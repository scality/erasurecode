package erasurecode

/*
#cgo pkg-config: erasurecode-1
#include <stdlib.h>
#include <liberasurecode/erasurecode.h>
#include <liberasurecode/erasurecode_helpers_ext.h>
#include <liberasurecode/erasurecode_postprocessing.h>
// shims to make working with frag arrays easier
char ** makeStrArray(int n) { return calloc(n, sizeof (char *)); }
void freeStrArray(char ** arr) { free(arr); }
void * getStrArrayItem(char ** arr, int idx) { return arr[idx]; }
char * getStrOffset(char *ptr, int offset) { return ptr + offset; }
void setStrArrayItem(char ** arr, int idx, unsigned char * val) { arr[idx] = (char *) val; }
// shims because the fragment headers use misaligned fields
uint64_t getOrigDataSize(struct fragment_header_s *header) { return header->meta.orig_data_size; }
uint32_t getBackendVersion(struct fragment_header_s *header) { return header->meta.backend_version; }
ec_backend_id_t getBackendID(struct fragment_header_s *header) { return header->meta.backend_id; }
uint32_t getECVersion(struct fragment_header_s *header) { return header->libec_version; }
int getHeaderSize() { return sizeof(struct fragment_header_s); }
int is_null(char *ptr) { return ptr == NULL; }

// decode_fast is used when we have all data fragment. Instead of doing a true decoding, we just
// reassemble all the fragment linearized in a buffer. This is mainly a copy of liberasurecode
// fragment_to_string function, excepted that we won't do any addionnal allocation
// k is the number of expected data fragment
// in is an array of all frags
// inlen is the array size
// dest is an already allocated buffer where data will be linearized
// destlen is the buffer size, and hence, the maximum number of bytes linearized
// outlen is a pointer containing the number of bytes really linearized in dest (always lower or equal to destlen)
// it returns dest if nothing went wrong, else null
char* decode_fast(int k, char **in, int inlen, char *dest, uint64_t destlen, uint64_t *outlen)
{
        int i;
        int curr_idx = 0;
        int orig_data_size = -1;
        char *frags[k];
        // cannot decode fastly
        if (inlen < k) {
            return NULL;
        }
        if (dest == NULL || outlen == NULL) {
            return NULL;
        }
		memset(frags, 0, sizeof(frags));
		// we start by iterating on all fragment, and ordering according to the fragment index
		// in the header all data fragment (fragment whose index is lower than k)
        for (i = 0; i < inlen && curr_idx != k; i++) {
                int index;
                int data_size;
                if (is_invalid_fragment_header((fragment_header_t*)in[i])) {
                    continue;
                }
                index = get_fragment_idx(in[i]);
                data_size = get_fragment_payload_size(in[i]);
                if (index < 0 || data_size < 0) {
                    continue;
                }
                if (orig_data_size < 0) {
                    orig_data_size = get_orig_data_size(in[i]);
                } else if(get_orig_data_size(in[i]) != orig_data_size) {
                    continue;
                }
                if (index >= k) {
                    continue;
                }
                if (frags[index] == NULL) {
                    curr_idx ++;
                    frags[index] = in[i];
                }
        }
		// if we don't have enough data fragment, we leave this function and will probably
		// fallback on a true deocodin function 
        if (curr_idx != k) {
            return NULL;
        }

		// compute how number of bytes will be linearized
        int tocopy = orig_data_size;
        int string_off = 0;
        *outlen = orig_data_size;
        if(destlen < orig_data_size) {
            *outlen = destlen;
            tocopy = destlen;
        }

		// copy in an ordered way all bytes of fragments in the buffer
        for (i = 0; i < k && tocopy > 0; i++) {
            char *f = get_data_ptr_from_fragment(frags[i]);
            int fsize = get_fragment_payload_size(frags[i]);
            int psize = tocopy > fsize ? fsize : tocopy;
            memcpy(dest + string_off, f, psize);
            tocopy -= psize;
            string_off += psize;
        }
        return dest;
}


struct encode_chunk_context {
  ec_backend_t instance; // backend instance
  char **datas;                  // the K datas
  char **codings;        // the M codings
  unsigned int number_of_subgroup; // number of subchunk in each K part
  unsigned int chunk_size;       // datasize of each subchunk
  unsigned int frags_len; // allocating size of each K+M objects
  int blocksize;          // k-bounds of data
  int k;
  int m;
};

// instead of encoding K blocks of data, we divide and subencode blocks of
// 'piecesize' bytes.
// desc  : liberasurecode handle
// data : the whole data to encode
// datalen : the datalen
// piecesize : the size of little blocks used for encoding
// ctx : contains informations such as the ECN schema (see below)
//
void encode_chunk_prepare(int desc,
                          char *data,
                          int datalen,
                          int piecesize,
                          struct encode_chunk_context *ctx)
{
  ctx->instance = liberasurecode_backend_instance_get_by_desc(desc);
  int i;
  const int k = ctx->instance->args.uargs.k;
  const int m = ctx->instance->args.uargs.m;

  // here we compute the number of (k) subgroup of 'piecesize' bytes we can create
  int block_size = piecesize * k;
  ctx->number_of_subgroup = datalen / block_size;
  if(ctx->number_of_subgroup * block_size != datalen) {
        ctx->number_of_subgroup++;
  }

  ctx->chunk_size = piecesize;

  ctx->k = k;
  ctx->m = m;

  ctx->datas     = calloc(ctx->k, sizeof(char*));
  ctx->codings   = calloc(ctx->m, sizeof(char*));
  ctx->frags_len = (sizeof(fragment_header_t) + piecesize) * ctx->number_of_subgroup;

  for (i = 0; i < ctx->k; ++i) {
    ctx->datas[i] = get_aligned_buffer16(ctx->frags_len);
  }

  for (i = 0; i < ctx->m; ++i) {
    ctx->codings[i] = get_aligned_buffer16(ctx->frags_len);
  }

}

// encode_chunk will encode a subset of the fragments data.
// It has to be considered that all the datas will not be divided in K blocks, but instead,
// they will be divided in N sub-blocks of K*chunksize fragments
// [-------------------data---------------------]
// {s1a | s1b | s1c | s1d}{s2a | s2b | s2c |d2d }
// fragment1 => [header1a|s1a|header2a|s2a]
// fragment2 => [header1b]s1b|header2b|s2b]
// fragment3 => [header1c]s1c|header2c|s2c]
// fragment4 => [header1d]s1d|header2d|s2d]
// this mapping will let be more efficient against get range pattern (when we are only interesting in
// having a small subset of data) especially when a whole fragment will be missing
int encode_chunk(int desc, char *data, int datalen, struct encode_chunk_context *ctx, int nth)
{
  ec_backend_t ec = ctx->instance;
  char *k_ref[ctx->k];
  char *m_ref[ctx->m];

  int one_cell_size = sizeof(fragment_header_t) + ctx->chunk_size;
  int i, ret;
  char const *const dataend = data + datalen;
  char *dataoffset = data + (ctx->k * nth) * ctx->chunk_size;
  if (nth >= ctx->number_of_subgroup) {
    return -1;
  }

  // Do the mapping as described above
  int tot_len_sum = 0;
  for (i = 0; i < ctx->k; i++) {
    char *ptr = &ctx->datas[i][nth * one_cell_size];
    fragment_header_t *hdr = (fragment_header_t*)ptr;
    hdr->magic = LIBERASURECODE_FRAG_HEADER_MAGIC;
        ptr = (char*) (hdr + 1);
        if(dataoffset < dataend) {
          int len_to_copy = ctx->chunk_size;
          if (len_to_copy > dataend - dataoffset) {
                  len_to_copy = dataend - dataoffset;
		  }
		  tot_len_sum += len_to_copy;
          memcpy(ptr, dataoffset, len_to_copy);
        }
        dataoffset += ctx->chunk_size;
    k_ref[i] = ptr;
  }

  for (i = 0; i < ctx->m; i++) {
    char *ptr = &ctx->codings[i][nth * one_cell_size];
    fragment_header_t *hdr = (fragment_header_t*)ptr;
    hdr->magic = LIBERASURECODE_FRAG_HEADER_MAGIC;
    ptr = (char*) (hdr + 1);
    m_ref[i] = ptr;
  }

  // do the true encoding according the backend used (isa-l, cauchy ....)
  ret = ec->common.ops->encode(ec->desc.backend_desc, k_ref, m_ref, ctx->chunk_size);
  if (ret < 0) {
      fprintf(stderr, "error encode ret = %d\n", ret);
      return -1;
  } else {
	// fill the headers with true len, fragment len ....
    ret = finalize_fragments_after_encode(ec, ctx->k, ctx->m, ctx->chunk_size, tot_len_sum, k_ref, m_ref);
    if (ret < 0) {
      fprintf(stderr, "error encode ret = %d\n", ret);
      return -1;
    }
  }
  return 0;
}
*/
import "C"

import (
	"bytes"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

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

// Params describe the encoding/decoding parameters
type Params struct {
	Name string
	K    int
	M    int
	W    int
	HD   int
}

// Backend is a wrapper of a backend descriptor of liberasurecode
type Backend struct {
	Params
	libecDesc C.int
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
	backend := Backend{params, 0}
	id, err := nameToID(backend.Name)
	if err != nil {
		return backend, err
	}
	desc := C.liberasurecode_instance_create(id, &C.struct_ec_args{
		k:  C.int(backend.K),
		m:  C.int(backend.M),
		w:  C.int(backend.W),
		hd: C.int(backend.HD),
		ct: C.CHKSUM_CRC32,
	})
	if desc < 0 {
		return backend, fmt.Errorf("instance_create() returned %v", errToName(-desc))
	}
	backend.libecDesc = desc

	// Workaround on init bug of Jerasure
	// Apparently, jerasure will crash if the
	// first encode is done concurrently with other encode.
	res, err := backend.Encode(bytes.Repeat([]byte("1"), 1000))

	if err != nil {
		backend.Close()
		return Backend{}, err
	}

	defer res.Free()

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
	Data [][]byte // Slice of []bytes ==> our K+N encoded fragments
	Free func()   // cleanup closure (to free C allocated data once it becomes useless)
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
		result[i] = (*[1 << 30]byte)(C.getStrArrayItem(dataFrags, C.int(i)))[:int(fragLength):int(fragLength)]
	}
	for i := 0; i < backend.M; i++ {
		result[i+backend.K] = (*[1 << 30]byte)(C.getStrArrayItem(parityFrags, C.int(i)))[:int(fragLength):int(fragLength)]
	}
	return &EncodeData{result, func() {
		C.liberasurecode_encode_cleanup(
			backend.libecDesc, dataFrags, parityFrags)
	}}, nil
}

// GetHeaderSize returns the size of the header written by liberasurecode before each chunks
func (backend *Backend) GetHeaderSize() int {
	r := int(C.getHeaderSize())
	return r
}

// EncodeMatrix encodes data in small subpart of chunkSize bytes
func (backend *Backend) EncodeMatrix(data []byte, chunkSize int) (*EncodeData, error) {
	var wg sync.WaitGroup
	var ctx C.struct_encode_chunk_context
	pData := (*C.char)(unsafe.Pointer(&data[0]))
	pDataLen := C.int(len(data))
	cChunkSize := C.int(chunkSize)

	C.encode_chunk_prepare(backend.libecDesc, pData, pDataLen, cChunkSize, &ctx)

	wg.Add(int(ctx.number_of_subgroup))
	var errCounter uint64
	for i := 0; i < int(ctx.number_of_subgroup); i++ {
		go func(nth int) {
			defer wg.Done()
			r := C.encode_chunk(backend.libecDesc, pData, pDataLen, &ctx, C.int(nth))
			if r < 0 {
				atomic.AddUint64(&errCounter, 1)
			}
		}(i)
	}
	wg.Wait()

	if errCounter != 0 {
		return &EncodeData{nil, func() {
				C.liberasurecode_encode_cleanup(
					backend.libecDesc, ctx.datas, ctx.codings)
			}},
			fmt.Errorf("error encoding chunk (%+v encoding failed)", errCounter)
	}
	result := make([][]byte, backend.K+backend.M)
	fragLen := ctx.frags_len
	for i := 0; i < backend.K; i++ {
		result[i] = (*[1 << 30]byte)(unsafe.Pointer(C.getStrArrayItem(ctx.datas, C.int(i))))[:int(C.int(fragLen)):int(C.int(fragLen))]

	}
	for i := 0; i < backend.M; i++ {
		result[i+backend.K] = (*[1 << 30]byte)(unsafe.Pointer(C.getStrArrayItem(ctx.codings, C.int(i))))[:int(C.int(fragLen)):int(C.int(fragLen))]
	}

	return &EncodeData{result, func() {
		C.liberasurecode_encode_cleanup(
			backend.libecDesc, ctx.datas, ctx.codings)
	}}, nil
}

// DecodeData is the structure returned by all Decode* function
// It contains a linearized data buffer and a Free closure (that can be null)
// that clean some C dynamically allocated objects
// If Free is not null, the closure should be used only when the Data is not needed anymore
type DecodeData struct {
	Data []byte
	Free func()
}

// DecodeMatrix decode all the subchunk of frags, and linearize data
func (backend *Backend) DecodeMatrix(frags [][]byte, piecesize int) (*DecodeData, error) {
	var data *C.char
	var wg sync.WaitGroup

	if len(frags) == 0 {
		return nil, errors.New("decoding requires at least one fragment")
	}

	cFrags := C.makeStrArray(C.int(len(frags)))
	defer C.freeStrArray(cFrags)
	for index, frag := range frags {
		C.setStrArrayItem(cFrags, C.int(index), (*C.uchar)(&frag[0]))
	}
	lenFrags1 := len(frags[0])
	lenBlock := piecesize + backend.GetHeaderSize()
	numBlock := lenFrags1 / lenBlock
	if numBlock*lenBlock != lenFrags1 {
		numBlock++
	}

	// allocate the C final Buffer
	data = (*C.char)(C.malloc(C.ulong(numBlock * piecesize * backend.K)))
	var mutex = &sync.Mutex{}
	errorNb := 0
	totLen := uint64(0)
	wg.Add(numBlock)

	for i := 0; i < numBlock; i++ {
		currBlock := i
		// launch goroutines, providing them a subrange of the final buffer so it can be used
		// in concurrency without need to lock it access
		go func(blockNr int) {
			cFrags := C.makeStrArray(C.int(len(frags)))
			defer C.freeStrArray(cFrags)
			// prepare the C array of pointer, respecting the offset in each fragments
			for index, frags := range frags {
				C.setStrArrayItem(cFrags, C.int(index), (*C.uchar)(&frags[blockNr*lenBlock]))
			}
			// try to decode fastly (if we have all data fragments), providing the good offset of the
			// linearized buffer, according the block number we are decoding
			var outlen C.uint64_t
			p := C.decode_fast(C.int(backend.K), cFrags, C.int(len(frags)),
				C.getStrOffset(data, C.int(blockNr*piecesize*backend.K)),
				C.ulong(piecesize*backend.K), &outlen)
			mutex.Lock()
			defer mutex.Unlock()
			if C.is_null(p) == C.int(1) {
				errorNb++
			} else {
				totLen += uint64(outlen)
			}
			wg.Done()
		}(currBlock)
	}
	wg.Wait()
	// if we got some issues, fallback on "slow" decoding
	if errorNb != 0 {
		C.free(unsafe.Pointer(data))
		return backend.decodeMatrixSlow(frags, piecesize)
	}

	// return our linearized data. Closure expect to free the C allocated data once
	// the DecodeData.Data will not be used anymore
	return &DecodeData{(*[1 << 30]byte)(unsafe.Pointer(data))[:int(totLen):int(totLen)],
			func() {
				C.liberasurecode_decode_cleanup(backend.libecDesc, data)
			}},
		nil

}

// decodeMatrixSlow is a fallback when something went wrong with decodeMatrix (especially when data part is missing)
func (backend *Backend) decodeMatrixSlow(frags [][]byte, piecesize int) (*DecodeData, error) {
	lenData := len(frags[0])
	blockSize := piecesize + backend.GetHeaderSize()
	blockNr := lenData / blockSize
	if blockNr*blockSize != lenData {
				blockNr++
	}
	data := make([]byte, 0)

	cellSize := piecesize + backend.GetHeaderSize()

	for i := 0; i < blockNr; i++ {
		var vect [][]byte
		for j := 0; j < len(frags); j++ {
			vect = append(vect, frags[j][i * cellSize : (i+1)* cellSize])
		}
		subdata, err := backend.Decode(vect)
		if err != nil {
			return nil, fmt.Errorf("error subdecoding %d cause =%v", i, err)
		}
		data = append(data, subdata.Data...)
		subdata.Free()
	}
	return &DecodeData{data, nil}, nil
}

// Decode is the general purpose decoding function (without sub-chunking)
func (backend *Backend) Decode(frags [][]byte) (*DecodeData, error) {
	var data *C.char
	var dataLength C.uint64_t
	if len(frags) == 0 {
		return nil, errors.New("decoding requires at least one fragment")
	}

	cFrags := C.makeStrArray(C.int(len(frags)))
	defer C.freeStrArray(cFrags)
	for index, frag := range frags {
		C.setStrArrayItem(cFrags, C.int(index), (*C.uchar)(&frag[0]))
	}

	if rc := C.liberasurecode_decode(
		backend.libecDesc, cFrags, C.int(len(frags)),
		C.uint64_t(len(frags[0])), C.int(1),
		&data, &dataLength); rc != 0 {
		return nil, fmt.Errorf("decode() returned %v", errToName(-rc))
	}
	runtime.KeepAlive(frags) // prevent frags from being GC-ed during decode

	return &DecodeData{(*[1 << 30]byte)(unsafe.Pointer(data))[:int(dataLength):int(dataLength)],
			func() {
				C.liberasurecode_decode_cleanup(backend.libecDesc, data)
			}},
		nil
}

// Reconstruct rebuild a missing fragment
func (backend *Backend) Reconstruct(frags [][]byte, fragIndex int) ([]byte, error) {
	if len(frags) == 0 {
		return nil, errors.New("reconstruction requires at least one fragment")
	}
	fragLength := len(frags[0])
	data := make([]byte, fragLength)
	pData := (*C.char)(unsafe.Pointer(&data[0]))

	cFrags := C.makeStrArray(C.int(len(frags)))
	defer C.freeStrArray(cFrags)
	for index, frag := range frags {
		C.setStrArrayItem(cFrags, C.int(index), (*C.uchar)(&frag[0]))
	}

	if rc := C.liberasurecode_reconstruct_fragment(
		backend.libecDesc, cFrags, C.int(len(frags)),
		C.uint64_t(len(frags[0])), C.int(fragIndex), pData); rc != 0 {
		return nil, fmt.Errorf("reconstruct_fragment() returned %v", errToName(-rc))
	}
	runtime.KeepAlive(frags) // prevent frags from being GC-ed during reconstruct
	return data, nil
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
	header := *(*C.struct_fragment_header_s)(unsafe.Pointer(&frag[0]))
	backendID := C.getBackendID(&header)
	return FragmentInfo{
		Index:               int(header.meta.idx),
		Size:                int(header.meta.size),
		BackendMetadataSize: int(header.meta.frag_backend_metadata_size),
		OrigDataSize:        uint64(C.getOrigDataSize(&header)),
		BackendID:           backendID,
		BackendName:         idToName(backendID),
		BackendVersion:      makeVersion(C.getBackendVersion(&header)),
		ErasureCodeVersion:  makeVersion(C.getECVersion(&header)),
		IsValid:             C.is_invalid_fragment_header((*C.fragment_header_t)(&header)) == 0,
	}
}
