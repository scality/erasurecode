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

// shims because the fragment headers use misaligned fields
uint64_t getOrigDataSize(struct fragment_header_s *header) { return header->meta.orig_data_size; }
uint32_t getBackendVersion(struct fragment_header_s *header) { return header->meta.backend_version; }
ec_backend_id_t getBackendID(struct fragment_header_s *header) { return header->meta.backend_id; }
uint32_t getECVersion(struct fragment_header_s *header) { return header->libec_version; }
int getHeaderSize() { return sizeof(struct fragment_header_s); }

// linearize is used when we have all data fragment. Instead of doing a true decoding, we just
// reassemble all the fragment linearized in a buffer. This is mainly a copy of liberasurecode
// fragment_to_string function, except that we won't do any addionnal allocation
//
// /!\ This function does not perform any header checksum validation.
// If fragments must be validated checks 'check_matrix_fragment'
//
// 'k' the number of data fragment used for the encoding.
// 'in' is an array of all data frags, in their index order.
// 'inlen' is the array size
// 'dest' is an already allocated buffer where data will be linearized
// 'destlen' is the buffer size, and hence, the maximum number of bytes linearized
// 'outlen' is a pointer containing the number of bytes really linearized in dest (always lower or equal to destlen)
// it returns dest if nothing went wrong, else null
char* linearize(int k, char **in, int inlen, char *dest, uint64_t destlen, uint64_t *outlen) {
    int i;
    int curr_idx = 0;
    int orig_data_size = -1;

    if (dest == NULL || outlen == NULL) {
        return NULL;
    }

    // The following perform small correctness checks before linearizing
    // the buffer
    int previous_idx = -1;
    for (i = 0; i < inlen; i++) {
        int index = get_fragment_idx(in[i]);

        if (orig_data_size < 0) {
            orig_data_size = get_orig_data_size(in[i]);
        } else if(get_orig_data_size(in[i]) != orig_data_size) {
            return NULL;
        }

        if (index >= k) {
            return NULL;
        }

        // Checks that the fragments are sorted.
        if (previous_idx > index) {
            return NULL;
        }
        previous_idx = index;
    }

    // compute the number of bytes needed for the output
    int tocopy = orig_data_size;
    int string_off = 0;
    *outlen = orig_data_size;
    if(destlen < orig_data_size) {
        *outlen = destlen;
        tocopy = destlen;
    }

    // copy in an ordered way all bytes of fragments in the buffer
    for (i = 0; i < inlen && tocopy > 0; i++) {
        char *f = get_data_ptr_from_fragment(in[i]);
        int fsize = get_fragment_payload_size(in[i]);
        int psize = tocopy > fsize ? fsize : tocopy;
        memcpy(dest + string_off, f, psize);
        tocopy -= psize;
        string_off += psize;
    }
    return dest;
}

bool check_matrix_fragment(char *frag, int frag_len, int piecesize) {
    size_t offset = 0;

    bool aligned = (frag_len % (piecesize + getHeaderSize())) == 0;
    if (!aligned) {
        return false;
    }

    while (offset < frag_len) {
        if (is_invalid_fragment_header((fragment_header_t*)&frag[offset])) {
            return false;
        }
        offset += piecesize + getHeaderSize();
    }

    return true;
}


struct encode_chunk_context {
    ec_backend_t instance; // backend instance
    char **datas;          // the K datas
    char **codings;        // the M codings
    unsigned int number_of_subgroup; // number of subchunk in each K part
    unsigned int chunk_size;         // datasize of each subchunk
    unsigned int frags_len; // allocating size of each K+M objects
    int blocksize;          // k-bounds of data
    int k;
    int m;
};

static inline void *alloc_data(size_t len) {
    void *buf;
    if (posix_memalign(&buf, 16, len) != 0) {
        return NULL;
    }
    memset(buf, 0, len);
    return buf;
}

static inline void dealloc_data(void *pt, size_t len) {
    free(pt);
}

// instead of encoding K blocks of data, we divide and subencode blocks of
// 'piecesize' bytes.
// 'desc'  : liberasurecode handle
// 'data' : the whole data to encode
// 'datalen' : the datalen
// 'piecesize' : the size of little blocks used for encoding
// 'ctx' : contains informations such as the ECN schema (see below)
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
        ctx->datas[i] = alloc_data(ctx->frags_len);
    }

    for (i = 0; i < ctx->m; ++i) {
        ctx->codings[i] = alloc_data(ctx->frags_len);
    }
}

// return real size of fragment header size
size_t get_fragment_header_size() {
    return sizeof(fragment_header_t);
}

int encode_chunk(int desc, char *data, int datalen, struct encode_chunk_context *ctx, int nth);

int encode_chunk_all(int desc, char *data, int datalen, struct encode_chunk_context *ctx, int max) {
    int i;
    for (i = 0; i < max ; i++) {
        int err = encode_chunk(desc, data, datalen, ctx, i);
        if (err != 0) {
            return err;
        }
    }
    return 0;
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
    }

    // fill the headers with true len, fragment len ....
    ret = finalize_fragments_after_encode(ec, ctx->k, ctx->m, ctx->chunk_size, tot_len_sum, k_ref, m_ref);
    if (ret < 0) {
        fprintf(stderr, "error encode ret = %d\n", ret);
        return -1;
    }
    return 0;
}

int my_liberasurecode_encode_cleanup(int desc,
    size_t len,
    char **encoded_data,
    char **encoded_parity)
{
    int i, k, m;

    ec_backend_t instance = liberasurecode_backend_instance_get_by_desc(desc);
    if (NULL == instance) {
        return -EBACKENDNOTAVAIL;
    }

    k = instance->args.uargs.k;
    m = instance->args.uargs.m;

    if (encoded_data) {
        for (i = 0; i < k; i++) {
            dealloc_data(encoded_data[i], len);
        }

        free(encoded_data);
    }

    if (encoded_parity) {
        for (i = 0; i < m; i++) {
            dealloc_data(encoded_parity[i], len);
        }
        free(encoded_parity);
    }

    return 0;
}

// Prepare memory, allocating stuff. Suitable for "buffermatrix": no data fragments allocated.
void encode_chunk_buffermatrix_prepare(int desc,
    char *data,
    int datalen,
    int piecesize,
    int frags_len,
    int number_of_subgroup,
    struct encode_chunk_context *ctx)
{
    ctx->instance = liberasurecode_backend_instance_get_by_desc(desc);
    int i;
    const int k = ctx->instance->args.uargs.k;
    const int m = ctx->instance->args.uargs.m;

    ctx->number_of_subgroup = number_of_subgroup;

    ctx->chunk_size = piecesize;

    ctx->k = k;
    ctx->m = m;

    ctx->codings   = calloc(ctx->m, sizeof(char*));
    ctx->frags_len = frags_len;

    for (i = 0; i < ctx->m; ++i) {
        ctx->codings[i] = alloc_data(ctx->frags_len);
    }
}

// Encode a chunk using a buffer matrix as an input
// Same as above with the twist that data is not copied and can be directly
static int encode_chunk_buffermatrix(int desc, char *data, int datalen, int nbFrags, struct encode_chunk_context *ctx, int nth)
{
    ec_backend_t ec = ctx->instance;
    char *k_ref[ctx->k];
    char *m_ref[ctx->m];
    int one_cell_size = sizeof(fragment_header_t) + ctx->chunk_size;
    int i, ret;
    int tot_len_sum = 0;

    if (nth >= ctx->number_of_subgroup) {
        return -1;
    }

    // Create the array of "data" fragments
    // No copy, just prepare the header
    for (i = 0 ; i < ctx->k ; i ++) {
        k_ref[i] = data + (nth + nbFrags * i) * one_cell_size;
        fragment_header_t *hdr = (fragment_header_t*)k_ref[i];
        hdr->magic = LIBERASURECODE_FRAG_HEADER_MAGIC;
        char *ptr = (char*) (hdr + 1);
        k_ref[i] = ptr;

        // Computes actual data in the fragment
        int size = datalen - (nth * ctx->k + i) * ctx->chunk_size;
        tot_len_sum += size > 0 ? (size > ctx->chunk_size ? ctx->chunk_size: size) : 0;
    }

    // "coding" fragments. Those ones are allocated above
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
    }

    ret = finalize_fragments_after_encode(ec, ctx->k, ctx->m, ctx->chunk_size, tot_len_sum, k_ref, m_ref);
    if (ret < 0) {
        fprintf(stderr, "error encode ret = %d\n", ret);
        return -1;
    }
    return 0;
}

// Helper function to compute everything in one go
int encode_chunk_buffermatrix_all(int desc, char *data, int datalen, int nbfrags, struct encode_chunk_context *ctx, int max) {
    int i;

    for (i = 0; i < max ; i++) {
        int err = encode_chunk_buffermatrix(desc, data, datalen, nbfrags, ctx, i);
        if (err != 0) {
            return err;
        }
    }
    return 0;
}

int my_liberasurecode_encode_buffermatrix_cleanup(int desc,
    size_t len,
    char **encoded_data,
    char **encoded_parity)
{
    ec_backend_t instance = liberasurecode_backend_instance_get_by_desc(desc);
    if (NULL == instance) {
        return -EBACKENDNOTAVAIL;
    }

    const int m = instance->args.uargs.m;

    if (encoded_parity) {
        int i;
        for (i = 0; i < m; i++) {
            dealloc_data(encoded_parity[i], len);
        }
    }
    free(encoded_parity);

    return 0;
}

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

func (p *pool) New(size int) (interface{}, []byte) {
	if size <= p.max {
		b := p.p.Get().(*bytes.Buffer)
		return b, b.Bytes()
	}
	// Should never happen
	return nil, make([]byte, size)
}

func (p *pool) Release(b interface{}) {
	if b != nil {
		p.p.Put(b)
	}
}

var globalPool = &pool{
	p: sync.Pool{
		New: func() interface{} {
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
				New: func() interface{} {
					return bytes.NewBuffer(make([]byte, params.MaxBlockSize))
				}},
			max: params.MaxBlockSize,
		}
	}

	// Workaround on init bug of Jerasure
	// Apparently, jerasure will crash if the
	// first encode is done concurrently with other encode.
	res, err := backend.Encode(bytes.Repeat([]byte("1"), 1000))

	if err != nil {
		backend.Close()
		return Backend{}, err
	}

	res.Free()

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
	}}, nil
}

// EncodeMatrixWithBufferMatrix encodes data in small subpart of chunkSize bytes
func (backend *Backend) EncodeMatrixWithBufferMatrix(bm *BufferMatrix, chunkSize int) (*EncodeData, error) {
	var wg sync.WaitGroup
	var ctx C.struct_encode_chunk_context

	data := bm.Bytes()
	dataLen := bm.Length()

	pData := (*C.char)(unsafe.Pointer(&data[0]))
	pDataLen := C.int(dataLen)
	cChunkSize := C.int(chunkSize)

	nbFrags := C.int(bm.SubGroups())

	C.encode_chunk_buffermatrix_prepare(backend.libecDesc, pData, pDataLen,
		cChunkSize, C.int(bm.FragLen()), nbFrags, &ctx)

	var errCounter uint32

	wg.Add(int(ctx.number_of_subgroup))

	for i := 0; i < int(ctx.number_of_subgroup); i++ {
		go func(nth int) {
			r := C.encode_chunk_buffermatrix(backend.libecDesc, pData, pDataLen, nbFrags, &ctx, C.int(nth))
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
			}},
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
	}}, nil
}

// EncodeMatrix encodes data in small subpart of chunkSize bytes
func (backend *Backend) EncodeMatrix(data []byte, chunkSize int) (*EncodeData, error) {
	var wg sync.WaitGroup
	var ctx C.struct_encode_chunk_context
	pData := (*C.char)(unsafe.Pointer(&data[0]))
	pDataLen := C.int(len(data))
	cChunkSize := C.int(chunkSize)

	C.encode_chunk_prepare(backend.libecDesc, pData, pDataLen, cChunkSize, &ctx)

	var errCounter uint32

	wg.Add(int(ctx.number_of_subgroup))

	for i := 0; i < int(ctx.number_of_subgroup); i++ {
		go func(nth int) {
			r := C.encode_chunk(backend.libecDesc, pData, pDataLen, &ctx, C.int(nth))
			if r < 0 {
				atomic.AddUint32(&errCounter, 1)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()

	if errCounter != 0 {
		return &EncodeData{nil, func() {
				C.my_liberasurecode_encode_cleanup(
					backend.libecDesc, C.size_t(ctx.frags_len), ctx.datas, ctx.codings)
			}},
			fmt.Errorf("error encoding chunk (%+v encoding failed)", errCounter)
	}
	result := make([][]byte, backend.K+backend.M)
	fragLen := ctx.frags_len
	for i := 0; i < backend.K; i++ {
		str := cGetArrayItem(ctx.datas, i)
		result[i] = (*[1 << 30]byte)(str)[:int(C.int(fragLen)):int(C.int(fragLen))]

	}
	for i := 0; i < backend.M; i++ {
		str := cGetArrayItem(ctx.codings, i)
		result[i+backend.K] = (*[1 << 30]byte)(str)[:int(C.int(fragLen)):int(C.int(fragLen))]
	}

	return &EncodeData{result, func() {
		C.my_liberasurecode_encode_cleanup(
			backend.libecDesc, C.size_t(ctx.frags_len), ctx.datas, ctx.codings)
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

func (backend *Backend) LinearizeMatrix(frags []ValidatedFragment, pieceSize int) (*DecodeData, error) {
	var wg sync.WaitGroup

	if len(frags) == 0 {
		return nil, errors.New("linearizing requires at least one fragment")
	}

	fragRangeLen := len(frags[0])
	chunkSize := pieceSize + backend.headerSize
	nrChunks := fragRangeLen / chunkSize
	if nrChunks*chunkSize != fragRangeLen {
		nrChunks++
	}

	/* Fragments are sorted beforehand with the index of the first chunk.
	   All chunks of a fragments share the same index. */
	fragsIndex := make([]int, len(frags))
	for i := 0; i < len(frags); i += 1 {
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

		lastDataFragIdxExcl += 1
		previousFragIdx = idx
	}
	fragsIndex = fragsIndex[:lastDataFragIdxExcl]

	dataB, data := backend.pool.New(nrChunks * pieceSize * len(fragsIndex))
	errorNb := uint32(0)
	totLen := uint64(0)
	wg.Add(nrChunks)

	for i := 0; i < nrChunks; i++ {
		// launch goroutines, providing them a subrange of the final buffer so it can be used
		// in concurrency without need to lock it access
		func(chunkIdx int) {
			cDataFrags := C.makeStrArray(C.int(len(fragsIndex)))
			// prepare the C array of pointer, respecting the offset in each fragments

			for i, idx := range fragsIndex {
				frag := frags[idx]
				cSetArrayItem(cDataFrags, i, (*C.char)(unsafe.Pointer(&frag[chunkIdx*chunkSize])))
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

	// if we got some issues, fallback on "slow" decoding
	if errorNb != 0 {
		// Release the previous buffer
		backend.pool.Release(dataB)
		return nil, errors.New("failed to linearize fragments")
	}

	return &DecodeData{
		data[:totLen:totLen],
		func() {
			backend.pool.Release(dataB)
		}}, nil
}

// DecodeMatrix tries to reconstruct the data fragments and returns the linearized data.
func (backend *Backend) DecodeMatrix(frags []ValidatedFragment, pieceSize int) (*DecodeData, error) {
	fragRangeLen := len(frags[0])
	chunkSize := pieceSize + backend.headerSize
	chunkNr := fragRangeLen / chunkSize
	if chunkNr*chunkSize != fragRangeLen {
		chunkNr++
	}

	dataB, data := backend.pool.New(chunkNr * pieceSize * backend.K)

	var totLen int64

	for i := 0; i < chunkNr; i++ {
		vect := make([][]byte, len(frags))
		for j := 0; j < len(frags); j++ {
			if len(frags[j]) != fragRangeLen {
				return nil, errors.New("invalid fragment len")
			}

			vect[j] = frags[j][i*chunkSize : (i+1)*chunkSize]
		}
		subdata, err := backend.Decode(vect)
		if err != nil {
			return nil, fmt.Errorf("error subdecoding %d cause =%v", i, err)
		}
		copy(data[totLen:], subdata.Data)
		totLen += int64(len(subdata.Data))
		subdata.Free()
	}

	return &DecodeData{data[:totLen:totLen], func() {
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
 * 		1. Each fragment range is always identical.
 *      2. When the requested range wraps around fragments all fragments
 *         are always queried.
 *
 * (1) is currently necessary to avoid querying multiple time the same
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
 *                     the heading and trailing is unecessary and could be
 *                     discarded in the ideal case.
 *
 * Perfect cases occur when the request span a single group (column):
 *
 *     p1 [- - - -]
 * 		  [-[*]- -]
 *     .. [-[*]- -]
 *     p4 [-[*]- -]
 *
 */
func (backend *Backend) GetRangeMatrix(startIncl, endIncl, pieceSize, fragSize int) *RangeMatrix {
	chunkSize := pieceSize + backend.headerSize
	groupSize := pieceSize * backend.K

	/* At this point we don't know what is the true payload size, but we
	   can at least check that it doesn't exceed the maximum payload that
	   this configuration can handle. */
	nrChunkByFrag := fragSize / chunkSize
	dataLenPerFrag := fragSize - nrChunkByFrag*backend.headerSize
	maxDataLen := dataLenPerFrag * backend.K
	if startIncl >= maxDataLen || endIncl >= maxDataLen || startIncl > endIncl {
		return nil
	}

	pieceStartIncl := startIncl / pieceSize
	pieceEndIncl := endIncl / pieceSize

	groupStartIncl := pieceStartIncl / backend.K
	groupEndIncl := pieceEndIncl / backend.K

	fragFirstIncl := pieceStartIncl % backend.K
	fragCount := (pieceEndIncl + 1 - pieceStartIncl)
	dataOffset := pieceStartIncl * pieceSize

	/* When wrapping around, we read the full groups. */
	if fragFirstIncl+fragCount > backend.K {
		fragFirstIncl = 0
		fragCount = backend.K
		dataOffset = groupStartIncl * groupSize
	}

	/* For each fragment, this is the minimum range to read -- including
	   the header -- to decode or repair the data. */
	inFragRangeStartIncl := groupStartIncl * chunkSize
	inFragRangeEndExcl := (groupEndIncl + 1) * chunkSize

	/* The output buffer only contains the data necessary to read the range,
	   and the requested range must be adjusted to be relative
	   to the output buffer which starts at 0.

	   Special care is needed whe the requested range wraps in the
	   fragments. In that case we degenerate to querying all groups of
	   all fragments (see (2) in the function's comment above). */
	linearizedRangeStartIncl := startIncl - dataOffset

	/* Decoding always works on a group boundary. */
	decodedRangeStartIncl := startIncl - groupStartIncl*groupSize

	return &RangeMatrix{
		ReqStartIncl:             startIncl,
		ReqEndIncl:               endIncl,
		FragFirstIncl:            fragFirstIncl,
		FragCount:                fragCount,
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

	return &DecodeData{(*[1 << 30]byte)(unsafe.Pointer(data))[:int(dataLength):int(dataLength)],
			func() {
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
func (backend *Backend) ReconstructMatrix(frags [][]byte, fragIndex int, chunksize int) (*DecodeData, error) {
	var wg sync.WaitGroup
	if len(frags) == 0 {
		return nil, errors.New("reconstruction requires at least one fragment")
	}

	fragLen := len(frags[0])
	blockSize := chunksize + backend.headerSize
	blockNr := fragLen / blockSize
	if blockNr*blockSize != fragLen {
		blockNr++
	}
	dlen := blockNr * blockSize
	dataB, data := backend.pool.New(dlen)

	cellSize := chunksize + backend.headerSize

	var errCounter uint32
	// TODO use goroutines here to leverage multicore computation
	wg.Add(blockNr)
	for i := 0; i < blockNr; i++ {
		go func(blocknr int) {
			vect := make([][]byte, len(frags))
			for j := 0; j < len(frags); j++ {
				vect[j] = frags[j][blocknr*cellSize : (blocknr+1)*cellSize]
			}
			if err := backend.reconstruct(vect, fragIndex, data[blocknr*blockSize:]); err != nil {
				atomic.AddUint32(&errCounter, 1)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	if errCounter != 0 {
		return nil, errors.New("sub reconstruction failed")
	}
	return &DecodeData{data[:dlen:dlen], func() {
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
