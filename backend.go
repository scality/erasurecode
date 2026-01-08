package erasurecode

/*
#cgo pkg-config: erasurecode-1
#include <stdlib.h>
#include "backend.h"
*/
import "C"
import (
	"bytes"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"
)

// Get the Nth value of a C string array
func cGetArrayItem(p **C.char, nth int) unsafe.Pointer {
	v1 := unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(nth)*unsafe.Sizeof(p))
	return unsafe.Pointer(*((**C.char)(v1)))
}

func cSetArrayItem(p **C.char, nth int, ptr *C.char) {
	v1 := unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(nth)*unsafe.Sizeof(p))
	*((**C.char)(v1)) = (ptr)
}

func fragmentHeaderSize() int {
	return int(C.get_fragment_header_size())
}

// Version describes the module version
type Version struct {
	Major    uint
	Minor    uint
	Revision uint
}

func (v Version) String() string {
	return fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Revision)
}

// Less compares two versions
func (v Version) Less(other Version) bool {
	if v.Major < other.Major {
		return true
	} else if v.Minor < other.Minor {
		return true
	} else if v.Revision < other.Revision {
		return true
	}
	return false
}

// GetVersion return a structure describing the version of the module
func GetVersion() Version {
	return makeVersion(C.liberasurecode_get_version())
}
func makeVersion(v C.uint32_t) Version {
	return Version{
		Major:    uint(v>>16) & 0xffff,
		Minor:    uint(v>>8) & 0xff,
		Revision: uint(v) & 0xff,
	}
}

// KnownBackends is a slice of all compatible backends
var KnownBackends = [...]string{
	"null",
	"jerasure_rs_vand",
	"jerasure_rs_cauchy",
	"flat_xor_hd",
	"isa_l_rs_vand",
	"shss",
	"liberasurecode_rs_vand",
	"isa_l_rs_cauchy",
	"libphazr",
}

// AvailableBackends fill a slice of all usable backends
func AvailableBackends() (avail []string) {
	for _, name := range KnownBackends {
		if BackendIsAvailable(name) {
			avail = append(avail, name)
		}
	}
	return
}

const ChecksumNone = C.CHKSUM_NONE
const ChecksumCrc32 = C.CHKSUM_CRC32
const ChecksumXxhash = C.CHKSUM_XXHASH

// Params describe the encoding/decoding parameters
type Params struct {
	Name         string
	K            int
	M            int
	W            int
	HD           int
	MaxBlockSize int
	Checksum     int
}

// Backend is a wrapper of a backend descriptor of liberasurecode
type Backend struct {
	Params
	libecDesc  C.int
	headerSize int
	pool       *pool
}

type pool struct {
	p   sync.Pool
	max int
}

// The max buffer size we can get from the pool.
// Assuming a splitSize around 2MiB or less
const maxBuffer int = 2 * 1024 * 1024

func (p *pool) New(size int) (any, []byte) {
	if size <= p.max {
		b := p.p.Get().(*bytes.Buffer)
		return b, b.Bytes()
	}
	// Should never happen
	return nil, make([]byte, size)
}

func (p *pool) Release(b any) {
	if b != nil {
		p.p.Put(b)
	}
}

var globalPool = &pool{
	p: sync.Pool{
		New: func() any {
			return bytes.NewBuffer(make([]byte, maxBuffer))
		}},
	max: maxBuffer,
}

// BackendIsAvailable check a backend availability
func BackendIsAvailable(name string) bool {
	id, err := nameToID(name)
	if err != nil {
		return false
	}
	return C.liberasurecode_backend_available(id) != 0
}

// InitBackend returns a backend descriptor according params provided
func InitBackend(params Params) (Backend, error) {
	if params.Checksum == 0 {
		params.Checksum = ChecksumXxhash
	}

	backend := Backend{params, 0, int(C.getHeaderSize()), nil}
	id, err := nameToID(backend.Name)
	if err != nil {
		return backend, err
	}
	desc := C.liberasurecode_instance_create(id, &C.struct_ec_args{
		k:  C.int(backend.K),
		m:  C.int(backend.M),
		w:  C.int(backend.W),
		hd: C.int(backend.HD),
		ct: C.ec_checksum_type_t(params.Checksum),
	})
	if desc < 0 {
		return backend, fmt.Errorf("instance_create() returned %v", errToName(-desc))
	}
	backend.libecDesc = desc

	// Create a pool to store the decoded data
	// Let's have a global pool for everything small enough
	// and create a dedicated one if this is too large to fit.
	if params.MaxBlockSize <= globalPool.max {
		backend.pool = globalPool
	} else {
		backend.pool = &pool{
			p: sync.Pool{
				New: func() any {
					return bytes.NewBuffer(make([]byte, params.MaxBlockSize))
				}},
			max: params.MaxBlockSize,
		}
	}

	return backend, nil
}

// Close cleans a backend descriptor
func (backend *Backend) Close() error {
	if backend.libecDesc == 0 {
		return errors.New("backend already closed")
	}
	if rc := C.liberasurecode_instance_destroy(backend.libecDesc); rc != 0 {
		return fmt.Errorf("instance_destroy() returned %v", errToName(-rc))
	}
	backend.libecDesc = 0
	return nil
}

// EncodeData is returned by all encode* functions
type EncodeData struct {
	Data         [][]byte // Slice of []bytes ==> our K+N encoded fragments
	Free         func()   // cleanup closure (to free C allocated data once it becomes useless)
	RealDataSize int64    // the real size of the data, without considering the padding
}

func (e EncodeData) DataLen() int64 {
	return e.RealDataSize
}
func (e *EncodeData) GetFragment(index int) []byte {
	if index < 0 || index >= len(e.Data) {
		return nil
	}
	return e.Data[index][:e.RealDataSize]
}

// Encode is the general purpose encoding function. It encodes data according
// backend params and returns an EncodeData structure containing the Fragments
func (backend *Backend) Encode(data []byte) (*EncodeData, error) {
	var dataFrags **C.char
	var parityFrags **C.char
	var fragLength C.uint64_t
	pData := (*C.char)(unsafe.Pointer(&data[0]))
	if rc := C.liberasurecode_encode(
		backend.libecDesc, pData, C.uint64_t(len(data)),
		&dataFrags, &parityFrags, &fragLength); rc != 0 {
		return nil, fmt.Errorf("encode() returned %v", errToName(-rc))
	}

	result := make([][]byte, backend.K+backend.M)
	for i := 0; i < backend.K; i++ {
		// Convert the data block into a slice without copying the data.
		// Note: the 1 << 30 is not really used, the slice is set to a length & a capacity.
		str := cGetArrayItem(dataFrags, i)
		result[i] = (*[1 << 30]byte)((str))[:int(fragLength):int(fragLength)]
	}
	for i := 0; i < backend.M; i++ {
		str := cGetArrayItem(parityFrags, i)
		result[i+backend.K] = (*[1 << 30]byte)(str)[:int(fragLength):int(fragLength)]
	}
	return &EncodeData{result, func() {
		C.my_liberasurecode_encode_cleanup(
			backend.libecDesc, C.size_t(fragLength), dataFrags, parityFrags)
	}, int64(fragLength)}, nil
}

// EncodeMatrixWithBufferMatrix encodes data in small subpart of chunkSize bytes
func (backend *Backend) EncodeMatrixWithBufferMatrix(bm *BufferMatrix, chunkSize int) (*EncodeData, error) {
	var wg sync.WaitGroup
	var ctx C.struct_encode_chunk_context
	var totLen int64

	data := bm.Bytes()
	dataLen := bm.Length()

	pData := (*C.char)(unsafe.Pointer(&data[0]))
	pDataLen := C.int(dataLen)
	cChunkSize := C.int(chunkSize)

	nbFrags := C.int(bm.SubGroups())

	// This prepares the context for encoding
	// It allocates the coding fragments (not the data fragments)
	C.encode_chunk_buffermatrix_prepare(backend.libecDesc, pData, pDataLen,
		cChunkSize, C.int(bm.FragLen()), nbFrags, &ctx)

	var errCounter uint32

	wg.Add(int(ctx.number_of_subgroup))

	for i := 0; i < int(ctx.number_of_subgroup); i++ {
		go func(nth int) {
			fragLen := C.size_t(chunkSize)
			// last subgroup has a different size
			if i == int(ctx.number_of_subgroup)-1 {
				fragLen = C.size_t(bm.FragLenLastSubGroup())
			}
			atomic.AddInt64(&totLen, int64(fragLen)+int64(backend.headerSize))
			r := C.encode_chunk_buffermatrix(backend.libecDesc, pData, pDataLen,
				nbFrags, &ctx, C.int(nth), fragLen)

			if r < 0 {
				atomic.AddUint32(&errCounter, 1)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()

	if errCounter != 0 {
		return &EncodeData{nil, func() {
				C.my_liberasurecode_encode_buffermatrix_cleanup(
					backend.libecDesc, C.size_t(ctx.frags_len), ctx.datas, ctx.codings)
			}, totLen},
			fmt.Errorf("error encoding chunk (%+v encoding failed)", errCounter)
	}
	result := make([][]byte, backend.K+backend.M)
	fragLen := ctx.frags_len
	flen := bm.FragLen()
	for i := 0; i < backend.K; i++ {
		result[i] = data[i*flen : (i+1)*flen]
	}

	for i := 0; i < backend.M; i++ {
		str := cGetArrayItem(ctx.codings, i)
		result[i+backend.K] = (*[1 << 30]byte)(str)[:int(C.int(fragLen)):int(C.int(fragLen))]
	}

	return &EncodeData{result, func() {
		runtime.KeepAlive(bm)
		C.my_liberasurecode_encode_buffermatrix_cleanup(
			backend.libecDesc, C.size_t(ctx.frags_len), ctx.datas, ctx.codings)
	}, totLen}, nil
}

// DecodeData is the structure returned by all Decode* function
// It contains a linearized data buffer and a Free closure (that can be null)
// that clean some C dynamically allocated objects
// If Free is not null, the closure should be used only when the Data is not needed anymore
type DecodeData struct {
	Data         []byte
	Free         func()
	RealDataSize int64
}

func (d DecodeData) DataLen() int64 {
	return d.RealDataSize
}

func (d *DecodeData) GetFragment() []byte {
	return d.Data[:d.RealDataSize]
}

// // bufPool is a pool of bytes.Buffer of max size maxBuffer.
// // It is up to the caller to wipe the buffer and make sure it
// // does overflow it
// var bufPool = sync.Pool{
// 	New: func() interface{} {
// 		return bytes.NewBuffer(make([]byte, maxBuffer))
// 	},
// }

// // getBuffer returns a []byte with a minimum size of `size`
// // @Note: it may be larger, you want to "reslice" it before.
// // One must call releaseBuffer on the first returned value
// // to release the buffer
// func getBuffer(size int) (interface{}, []byte) {
// 	if size < maxBuffer {
// 		b := bufPool.Get().(*bytes.Buffer)
// 		return b, b.Bytes()
// 	}
// 	return nil, make([]byte, size)
// }

// // releaseBuffer returns the underneath buffer to the pool
// // It is NOT safe to use the associated []byte array after releasing
// // it. Passing `nil` is safe.
// func releaseBuffer(buf interface{}) {
// 	if buf != nil {
// 		bufPool.Put(buf)
// 	}
// }

type RawFragment = []byte
type ValidatedFragment = []byte

func (backend *Backend) ValidateFragmentMatrix(frag RawFragment, pieceSize int) bool {
	result := C.check_matrix_fragment((*C.char)(unsafe.Pointer(&frag[0])), C.int(len(frag)), C.int(pieceSize))
	if result != C.bool(true) {
		return false
	}

	return true
}

type ChunkInfo struct {
	ChunkSize int
	NrChunk   int
}

func (backend *Backend) ChunkInfo(fragRangeLen int, pieceSize int) ChunkInfo {
	chunkSize := pieceSize + backend.headerSize
	nrChunks := fragRangeLen / chunkSize
	if nrChunks*chunkSize != fragRangeLen {
		nrChunks++
	}
	return ChunkInfo{
		ChunkSize: chunkSize,
		NrChunk:   nrChunks,
	}
}

func (backend *Backend) LinearizeMatrix(frags []ValidatedFragment, pieceSize int) (*DecodeData, error) {
	var wg sync.WaitGroup

	if len(frags) == 0 {
		return nil, errors.New("linearizing requires at least one fragment")
	}

	fragRangeLen := len(frags[0])
	chunkInfo := backend.ChunkInfo(fragRangeLen, pieceSize)

	/* Fragments are sorted beforehand with the index of the first chunk.
	   All chunks of a fragments share the same index. */
	fragsIndex := make([]int, len(frags))
	for i := range frags {
		fragsIndex[i] = i
	}

	sort.Slice(fragsIndex, func(i, j int) bool {
		lhs := frags[fragsIndex[i]]
		rhs := frags[fragsIndex[j]]

		var lhsIdx, rhsIdx C.int
		lhsIdx = C.get_fragment_idx((*C.char)(unsafe.Pointer(&lhs[0])))
		rhsIdx = C.get_fragment_idx((*C.char)(unsafe.Pointer(&rhs[0])))

		return lhsIdx < rhsIdx
	})

	/* There is no reconstruction that can happen.
	   All coding fragments must be ignored */
	lastDataFragIdxExcl := 0
	previousFragIdx := -1
	for lastDataFragIdxExcl < len(fragsIndex) {
		frag := frags[fragsIndex[lastDataFragIdxExcl]]
		idx := int(C.get_fragment_idx((*C.char)(unsafe.Pointer(&frag[0]))))

		if idx < 0 {
			return nil, errors.New("invalid fragment index")
		}

		if idx >= backend.K {
			break
		}

		if len(frags[0]) != fragRangeLen {
			return nil, errors.New("invalid fragment len")
		}

		if previousFragIdx > 0 && (idx-previousFragIdx) != 1 {
			/* Fragments are not contiguous. This functions doesn't supports
			   gaps. */
			return nil, errors.New("gaps in the provided fragments")
		}

		lastDataFragIdxExcl++
		previousFragIdx = idx
	}
	fragsIndex = fragsIndex[:lastDataFragIdxExcl]

	dataB, data := backend.pool.New(chunkInfo.NrChunk * pieceSize * len(fragsIndex))
	errorNb := uint32(0)
	totLen := uint64(0)
	wg.Add(chunkInfo.NrChunk)

	for i := 0; i < chunkInfo.NrChunk; i++ {
		// launch goroutines, providing them a subrange of the final buffer so it can be used
		// in concurrency without need to lock it access
		go func(chunkIdx int) {
			cDataFrags := C.makeStrArray(C.int(len(fragsIndex)))
			// prepare the C array of pointer, respecting the offset in each fragments

			for i, idx := range fragsIndex {
				frag := frags[idx]
				cSetArrayItem(cDataFrags, i, (*C.char)(unsafe.Pointer(&frag[chunkIdx*chunkInfo.ChunkSize])))
			}
			// try to decode fastly (if we have all data fragments), providing the good offset of the
			// linearized buffer, according the block number we are decoding
			var outlen C.uint64_t
			p := C.linearize(C.int(backend.K), cDataFrags, C.int(len(fragsIndex)),
				(*C.char)(unsafe.Pointer(&data[chunkIdx*pieceSize*len(fragsIndex)])),
				C.uint64_t(pieceSize*backend.K), &outlen)

			if p == nil {
				atomic.AddUint32(&errorNb, 1)
			} else {
				atomic.AddUint64(&totLen, uint64(outlen))
			}

			C.freeStrArray(cDataFrags)
			wg.Done()
		}(i)
	}
	wg.Wait()

	/* Tasks above which call into C.linearize keep a pointer of each
	   fragment to perform their computation. When all goroutines are in the
	   ffi call, there is no outstanding reference to frags. This ensure
	   that this array (and each frag that it references) do not get GC until
	   all tasks completed. */
	runtime.KeepAlive(frags)

	if errorNb != 0 {
		// Release the previous buffer
		backend.pool.Release(dataB)
		return nil, errors.New("failed to linearize fragments")
	}

	return &DecodeData{
		Data:         data[:totLen:totLen],
		RealDataSize: int64(totLen),
		Free: func() {
			backend.pool.Release(dataB)
		}}, nil
}

// DecodeMatrix tries to reconstruct the data fragments and returns the linearized data.
func (backend *Backend) DecodeMatrix(frags []ValidatedFragment, pieceSize int) (*DecodeData, error) {
	if len(frags) == 0 {
		return nil, errors.New("decoding requires at least one fragment")
	}

	fragRangeLen := len(frags[0])
	chunkInfo := backend.ChunkInfo(fragRangeLen, pieceSize)

	dataB, data := backend.pool.New(chunkInfo.NrChunk * pieceSize * backend.K)

	for i := range frags {
		if len(frags[i]) != fragRangeLen {
			return nil, errors.New("invalid fragment len")
		}
	}

	var totLen int64
	for i := 0; i < chunkInfo.NrChunk; i++ {
		vect := make([][]byte, len(frags))
		nextBound := min((i+1)*chunkInfo.ChunkSize, fragRangeLen)
		for j := range frags {
			vect[j] = frags[j][i*chunkInfo.ChunkSize : nextBound]
		}

		subdata, err := backend.Decode(vect)
		if err != nil {
			return nil, fmt.Errorf("error subdecoding %d cause =%v", i, err)
		}
		copy(data[totLen:], subdata.Data)
		totLen += int64(len(subdata.Data))
		subdata.Free()
	}

	return &DecodeData{
		Data:         data[:totLen:totLen],
		RealDataSize: int64(totLen),
		Free: func() {
			backend.pool.Release(dataB)
		}}, nil
}

// RangeMatrix describes information needed to decode a range of encoded frags
type RangeMatrix struct {
	ReqStartIncl int
	ReqEndIncl   int

	/* The fragments that the range spans.  */
	FragFirstIncl int
	FragCount     int

	/* The range in each fragment to be queried satisfy the requested range. */
	InFragRangeStartIncl int
	InFragRangeEndExcl   int

	/* The requested range relative to the decoded buffer. */
	LinearizedRangeStartIncl int
	DecodedRangeStartIncl    int
}

/*
 * Returns the ranges to read a matrix encoded data. This function tries
 * to minimize the number of request to perform depending on the requested
 * range.
 *
 * There are a few design choices that make the result not always obvious.
 *      1. Each fragment range is always identical.
 *      2. When the requested range wraps around fragments all fragments
 *         are always queried.
 *
 * (1) is currently necessary to avoid querying multiple times the same
 * fragment in case of failures (to reconstruct the data). This prefer
 * performing the minimum amount of IO requests, instead of reading the minimum
 * amount of data. We could lift this constraint if the caller would stream
 * group at a time but that would require the backend to have matching
 * alignement constraints.
 *
 * (2) could also be lifted. This is currently done to avoid changing the way
 * the current decoding is performed. To work, it currently requires consecutive
 * fragments. We then can't leave gaps in fragments. For example if
 * we take a erasure code with 4 data fragments of 4 chunks,
 * with the requested range represented by a '*' and the resulting fragment
 * ranges by '[]':
 *
 *     p1 [-[- *]-]    The request here start at the end of the 2nd chunk in p4
 *        [-[- *]-]    then wraps around in the subsequent chunks in p1 and p2.
 *     .. [-[- -]-]
 *     p4 [-[* -]-]    When this occurs, we will still query, p4 just to decode
 *                     the relevant requested range. The decoded buffer
 *                     will look like [p1 p2 p3 p4(*) p1(*) p2(*) p3 p4].
 *                     Chunks not marked with a '*' are discarded. Note how
 *                     the heading and trailing is unnecessary and could be
 *                     discarded in the ideal case.
 *
 * Perfect cases occur when the request span a single group (column):
 *
 *     p1 [- - - -]
 *        [-[*]- -]
 *     .. [-[*]- -]
 *     p4 [-[*]- -]
 *
 */
func (backend *Backend) GetRangeMatrix(startIncl, endIncl, cellDataSize, fragSize int) *RangeMatrix {
	nrColumns := backend.K
	cellSize := cellDataSize + backend.headerSize
	lineSize := cellDataSize * nrColumns

	/* At this point we don't know what is the true payload size, but we
	   can at least check that it doesn't exceed the maximum payload that
	   this configuration can handle. */
	nrLines := fragSize / cellSize
	//
	if nrLines*cellSize < fragSize {
		nrLines++
	}

	/* Inside a fragment (ie, inside a column), what is the
	   stored amount of data? */
	fragDataSize := fragSize - nrLines*backend.headerSize

	maxDataSize := fragDataSize * nrColumns

	if startIncl >= maxDataSize || endIncl >= maxDataSize || startIncl > endIncl {
		return nil
	}

	/* convert cells indices to (x,y) indices */

	/* Considering our (x,y) matrix as a single row, what
	   are the indices of the start and end of the range we are interested in? */
	idxStart := startIncl / cellDataSize
	idxEnd := endIncl / cellDataSize

	/* as we have the indices of the first cell and the last cell,
	 * we can derive the indices of the first line and the last line
	 * Based on this first line and last line, we can deduce
	 * the amount of data to read in each fragment.
	 */
	lineStart := idxStart / nrColumns
	lineEnd := idxEnd / nrColumns

	/*
	 * as we have the indices of the first cell and the last cell,
	 * we can compute the first column (e.g the first fragment)
	 * where to start the read
	 */
	columnStart := idxStart % nrColumns

	nrCellsToRead := (idxEnd + 1 - idxStart)
	dataOffset := idxStart * cellDataSize

	totalLines := (maxDataSize + lineSize - 1) / lineSize
	isLastStripe := (lineEnd == totalLines-1)

	/* When wrapping around, we read the full groups. */
	if columnStart+nrCellsToRead > nrColumns || isLastStripe {
		columnStart = 0
		nrCellsToRead = nrColumns
		dataOffset = lineStart * lineSize
	}

	/* For each fragment, this is the minimum range to read -- including
	   the header -- to decode or repair the data. */
	inFragRangeStartIncl := lineStart * cellSize
	inFragRangeEndExcl := (lineEnd + 1) * cellSize

	/* The output buffer only contains the data necessary to read the range,
	   and the requested range must be adjusted to be relative
	   to the output buffer which starts at 0.

	   Special care is needed whe the requested range wraps in the
	   fragments. In that case we degenerate to querying all groups of
	   all fragments (see (2) in the function's comment above). */
	linearizedRangeStartIncl := startIncl - dataOffset

	/* Decoding always works on a group boundary. */
	decodedRangeStartIncl := startIncl - lineStart*lineSize

	return &RangeMatrix{
		ReqStartIncl:             startIncl,
		ReqEndIncl:               endIncl,
		FragFirstIncl:            columnStart,
		FragCount:                nrCellsToRead,
		InFragRangeStartIncl:     inFragRangeStartIncl,
		InFragRangeEndExcl:       inFragRangeEndExcl,
		DecodedRangeStartIncl:    decodedRangeStartIncl,
		LinearizedRangeStartIncl: linearizedRangeStartIncl,
	}
}

// Decode is the general purpose decoding function (without sub-chunking)
func (backend *Backend) Decode(frags [][]byte) (*DecodeData, error) {
	var data *C.char
	var dataLength C.uint64_t
	if len(frags) == 0 {
		return nil, errors.New("decoding requires at least one fragment")
	}

	cFrags := C.makeStrArray(C.int(len(frags)))

	for index, frag := range frags {
		ptr := unsafe.Pointer(&frag[0])
		cSetArrayItem(cFrags, index, (*C.char)(ptr))
	}

	if rc := C.liberasurecode_decode(
		backend.libecDesc, cFrags, C.int(len(frags)),
		C.uint64_t(len(frags[0])), C.int(1),
		&data, &dataLength); rc != 0 {
		C.freeStrArray(cFrags)
		return nil, fmt.Errorf("decode() returned %v", errToName(-rc))
	}

	runtime.KeepAlive(frags) // prevent frags from being GC-ed during decode
	C.freeStrArray(cFrags)

	return &DecodeData{
			Data:         (*[1 << 30]byte)(unsafe.Pointer(data))[:int(dataLength):int(dataLength)],
			RealDataSize: int64(dataLength),
			Free: func() {
				C.liberasurecode_decode_cleanup(backend.libecDesc, data)
			}},
		nil
}

func (backend *Backend) reconstruct(frags [][]byte, fragIndex int, data []byte) error {
	if len(frags) == 0 {
		return errors.New("reconstruction requires at least one fragment")
	}

	pData := (*C.char)(unsafe.Pointer(&data[0]))

	cFrags := C.makeStrArray(C.int(len(frags)))

	for index, frag := range frags {
		ptr := unsafe.Pointer(&frag[0])
		cSetArrayItem(cFrags, index, (*C.char)(ptr))
	}

	if rc := C.liberasurecode_reconstruct_fragment(
		backend.libecDesc, cFrags, C.int(len(frags)),
		C.uint64_t(len(frags[0])), C.int(fragIndex), pData); rc != 0 {
		C.freeStrArray(cFrags)
		return fmt.Errorf("reconstruct_fragment() returned %v", errToName(-rc))
	}
	C.freeStrArray(cFrags)
	runtime.KeepAlive(frags) // prevent frags from being GC-ed during reconstruct
	return nil
}

// Reconstruct rebuild a missing fragment
func (backend *Backend) Reconstruct(frags [][]byte, fragIndex int) ([]byte, error) {
	if len(frags) == 0 {
		return nil, errors.New("reconstruction requires at least one fragment")
	}
	fragLength := len(frags[0])
	data := make([]byte, fragLength)

	if err := backend.reconstruct(frags, fragIndex, data); err != nil {
		return nil, err
	}
	return data, nil
}

// ReconstructMatrix is a really not optimized yet reconstruction of a frag containing subchunking
func (backend *Backend) ReconstructMatrix(frags [][]byte, fragIndex int, pieceSize int) (*DecodeData, error) {
	var wg sync.WaitGroup
	if len(frags) == 0 {
		return nil, errors.New("reconstruction requires at least one fragment")
	}

	fragLen := len(frags[0])
	chunkSize := pieceSize + backend.headerSize
	chunkNr := fragLen / chunkSize
	if chunkNr*chunkSize != fragLen {
		chunkNr++
	}
	dlen := chunkNr * chunkSize
	dataB, data := backend.pool.New(dlen)

	var errCounter uint32
	var totLen int64
	// TODO use goroutines here to leverage multicore computation
	wg.Add(chunkNr)
	for i := 0; i < chunkNr; i++ {
		go func(chunkIdx int) {
			vect := make([][]byte, len(frags))
			for j := range frags {
				length := min(len(frags[j]), (chunkIdx+1)*chunkSize)
				vect[j] = frags[j][chunkIdx*chunkSize : length]
			}
			atomic.AddInt64(&totLen, int64(len(vect[0])))
			if err := backend.reconstruct(vect, fragIndex, data[chunkIdx*chunkSize:]); err != nil {
				atomic.AddUint32(&errCounter, 1)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	if errCounter != 0 {
		return nil, errors.New("sub reconstruction failed")
	}

	return &DecodeData{
		Data:         data[:totLen:totLen],
		RealDataSize: int64(totLen),
		Free: func() {
			backend.pool.Release(dataB)
		}}, nil
}

// IsInvalidFragment is a wrapper on C implementation
// it checks that a fragment is not valid, according its header
func (backend *Backend) IsInvalidFragment(frag []byte) bool {
	pData := (*C.char)(unsafe.Pointer(&frag[0]))
	return 1 == C.is_invalid_fragment(backend.libecDesc, pData)
}

// FragmentInfo is a wrapper of fragment_header C struct
type FragmentInfo struct {
	Index               int
	Size                int
	BackendMetadataSize int
	OrigDataSize        uint64
	BackendID           C.ec_backend_id_t
	BackendName         string
	BackendVersion      Version
	ErasureCodeVersion  Version
	IsValid             bool
}

// GetFragmentInfo is the wrapper of the C implementation
func GetFragmentInfo(frag []byte) FragmentInfo {
	header := (*C.struct_fragment_header_s)(unsafe.Pointer(&frag[0]))
	backendID := C.getBackendID(header)
	return FragmentInfo{
		Index:               int(header.meta.idx),
		Size:                int(header.meta.size),
		BackendMetadataSize: int(header.meta.frag_backend_metadata_size),
		OrigDataSize:        uint64(C.getOrigDataSize(header)),
		BackendID:           backendID,
		BackendName:         idToName(backendID),
		BackendVersion:      makeVersion(C.getBackendVersion(header)),
		ErasureCodeVersion:  makeVersion(C.getECVersion(header)),
		IsValid:             C.is_invalid_fragment_header((*C.fragment_header_t)(header)) == 0,
	}
}
