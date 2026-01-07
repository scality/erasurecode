#include <liberasurecode/erasurecode.h>
#include <liberasurecode/erasurecode_helpers_ext.h>
#include <liberasurecode/erasurecode_postprocessing.h>
#include <stdint.h>
#include <stdlib.h>
#include "backend.h"

// shims to make working with frag arrays easier
char **makeStrArray(int n) { return calloc(n, sizeof(char *)); }

void freeStrArray(char **arr) { free(arr); }

uint64_t getOrigDataSize(struct fragment_header_s *header) {
  return header->meta.orig_data_size;
}
uint32_t getBackendVersion(struct fragment_header_s *header) {
  return header->meta.backend_version;
}
ec_backend_id_t getBackendID(struct fragment_header_s *header) {
  return header->meta.backend_id;
}
uint32_t getECVersion(struct fragment_header_s *header) {
  return header->libec_version;
}
int getHeaderSize() { return sizeof(struct fragment_header_s); }

// shims because the fragment headers use misaligned fields

// linearize is used when we have all data fragment. Instead of doing a true
// decoding, we just reassemble all the fragment linearized in a buffer. This is
// mainly a copy of liberasurecode fragment_to_string function, except that we
// won't do any addionnal allocation
//
// /!\ This function does not perform any header checksum validation.
// If fragments must be validated checks 'check_matrix_fragment'
//
// 'k' the number of data fragment used for the encoding.
// 'in' is an array of all data frags, in their index order.
// 'inlen' is the array size
// 'dest' is an already allocated buffer where data will be linearized
// 'destlen' is the buffer size, and hence, the maximum number of bytes
// linearized 'outlen' is a pointer containing the number of bytes really
// linearized in dest (always lower or equal to destlen) it returns dest if
// nothing went wrong, else null
char *linearize(int k, char **in, int inlen, char *dest, uint64_t destlen,
                uint64_t *outlen) {
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
    } else if (get_orig_data_size(in[i]) != orig_data_size) {
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
  if (destlen < orig_data_size) {
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

  while (offset < frag_len) {
    if (is_invalid_fragment_header((fragment_header_t *)&frag[offset])) {
      return false;
    }
    offset += piecesize + getHeaderSize();
  }

  return true;
}

static inline void *alloc_data(size_t len) {
  void *buf;
  if (posix_memalign(&buf, 16, len) != 0) {
    return NULL;
  }
  memset(buf, 0, len);
  return buf;
}

static inline void dealloc_data(void *pt, size_t len) { free(pt); }

// instead of encoding K blocks of data, we divide and subencode blocks of
// 'piecesize' bytes.
// 'desc'  : liberasurecode handle
// 'data' : the whole data to encode
// 'datalen' : the datalen
// 'piecesize' : the size of little blocks used for encoding
// 'ctx' : contains informations such as the ECN schema (see below)
//
void encode_chunk_prepare(int desc, char *data, int datalen, int piecesize,
                          struct encode_chunk_context *ctx) {
  ctx->instance = liberasurecode_backend_instance_get_by_desc(desc);
  int i;
  const int k = ctx->instance->args.uargs.k;
  const int m = ctx->instance->args.uargs.m;

  // here we compute the number of (k) subgroup of 'piecesize' bytes we can
  // create
  int block_size = piecesize * k;
  ctx->number_of_subgroup = datalen / block_size;
  if (ctx->number_of_subgroup * block_size != datalen) {
    ctx->number_of_subgroup++;
  }

  // Note: last chunk may be smaller than piecesize
  ctx->chunk_size = piecesize;

  ctx->k = k;
  ctx->m = m;

  ctx->datas = calloc(ctx->k, sizeof(char *));
  ctx->codings = calloc(ctx->m, sizeof(char *));
  ctx->frags_len =
      (sizeof(fragment_header_t) + piecesize) * ctx->number_of_subgroup;

  for (i = 0; i < ctx->k; ++i) {
    ctx->datas[i] = alloc_data(ctx->frags_len);
  }

  for (i = 0; i < ctx->m; ++i) {
    ctx->codings[i] = alloc_data(ctx->frags_len);
  }
}

// return real size of fragment header size
size_t get_fragment_header_size() { return sizeof(fragment_header_t); }

int encode_chunk(int desc, char *data, int datalen,
                 struct encode_chunk_context *ctx, int nth);

int encode_chunk_all(int desc, char *data, int datalen,
                     struct encode_chunk_context *ctx, int max) {
  int i;
  for (i = 0; i < max; i++) {
    int err = encode_chunk(desc, data, datalen, ctx, i);
    if (err != 0) {
      return err;
    }
  }
  return 0;
}

// encode_chunk will encode a subset of the fragments data.
// It has to be considered that all the datas will not be divided in K blocks,
// but instead, they will be divided in N sub-blocks of K*chunksize fragments
// [-------------------data---------------------]
// {s1a | s1b | s1c | s1d}{s2a | s2b | s2c |d2d }
// fragment1 => [header1a|s1a|header2a|s2a]
// fragment2 => [header1b]s1b|header2b|s2b]
// fragment3 => [header1c]s1c|header2c|s2c]
// fragment4 => [header1d]s1d|header2d|s2d]
// this mapping will let be more efficient against get range pattern (when we
// are only interesting in having a small subset of data) especially when a
// whole fragment will be missing
int encode_chunk(int desc, char *data, int datalen,
                 struct encode_chunk_context *ctx, int nth) {
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
    fragment_header_t *hdr = (fragment_header_t *)ptr;
    hdr->magic = LIBERASURECODE_FRAG_HEADER_MAGIC;
    ptr = (char *)(hdr + 1);
    if (dataoffset < dataend) {
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
    fragment_header_t *hdr = (fragment_header_t *)ptr;
    hdr->magic = LIBERASURECODE_FRAG_HEADER_MAGIC;
    ptr = (char *)(hdr + 1);
    m_ref[i] = ptr;
  }

  // do the true encoding according the backend used (isa-l, cauchy ....)
  ret = ec->common.ops->encode(ec->desc.backend_desc, k_ref, m_ref,
                               ctx->chunk_size);
  if (ret < 0) {
    return -1;
  }

  // fill the headers with true len, fragment len ....
  ret = finalize_fragments_after_encode(ec, ctx->k, ctx->m, ctx->chunk_size,
                                        tot_len_sum, k_ref, m_ref);
  if (ret < 0) {
    return -1;
  }
  return 0;
}

int my_liberasurecode_encode_cleanup(int desc, size_t len, char **encoded_data,
                                     char **encoded_parity) {
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

// Prepare memory, allocating stuff.
// Suitable for "buffermatrix": no data fragments allocated.
// Will also init chunk_size and number_of_subgroup
void encode_chunk_buffermatrix_prepare(int desc, char *data, int datalen,
                                       int piecesize, int frags_len,
                                       int number_of_subgroup,
                                       struct encode_chunk_context *ctx) {
  ctx->instance = liberasurecode_backend_instance_get_by_desc(desc);
  int i;
  const int k = ctx->instance->args.uargs.k;
  const int m = ctx->instance->args.uargs.m;

  ctx->number_of_subgroup = number_of_subgroup;

  // Note: last subgroup may be smaller than the others
  ctx->chunk_size = piecesize;

  ctx->k = k;
  ctx->m = m;

  ctx->codings = calloc(ctx->m, sizeof(char *));
  ctx->frags_len = frags_len;

  for (i = 0; i < ctx->m; ++i) {
    ctx->codings[i] = alloc_data(ctx->frags_len);
  }
}

// Encode a chunk using a buffer matrix as an input
// Same as above with the twist that data is not copied and can be directly
int encode_chunk_buffermatrix(int desc,
                              char *data,
                              int datalen,
                              int nbFrags,
                              struct encode_chunk_context *ctx,
                              int nth,
                              size_t fraglen) {
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
  for (i = 0; i < ctx->k; i++) {
    k_ref[i] = data + (nth + nbFrags * i) * one_cell_size;
    fragment_header_t *hdr = (fragment_header_t *)k_ref[i];
    hdr->magic = LIBERASURECODE_FRAG_HEADER_MAGIC;
    char *ptr = (char *)(hdr + 1);
    k_ref[i] = ptr;

    // Computes actual data in the fragment
    // If we are at the end of the data, we may have a smaller
    // fragment than the others. fraglen will take of that.
    int size = datalen - ((nth * ctx->k * ctx->chunk_size) + (i * fraglen));
    tot_len_sum += size > 0 ? (size > fraglen ? fraglen : size) : 0;
  }


  // "coding" fragments. Those ones are allocated above
  for (i = 0; i < ctx->m; i++) {
    char *ptr = &ctx->codings[i][nth * one_cell_size];
    fragment_header_t *hdr = (fragment_header_t *)ptr;
    hdr->magic = LIBERASURECODE_FRAG_HEADER_MAGIC;
    ptr = (char *)(hdr + 1);
    m_ref[i] = ptr;
  }


  // do the true encoding according the backend used (isa-l, cauchy ....)
  ret = ec->common.ops->encode(ec->desc.backend_desc, k_ref, m_ref,
                               fraglen);

  if (ret < 0) {
    return -1;
  }

  ret = finalize_fragments_after_encode(ec, ctx->k, ctx->m, fraglen,
                                        tot_len_sum, k_ref, m_ref);
  if (ret < 0) {
    return -1;
  }
  return 0;
}

// Helper function to compute everything in one go
int encode_chunk_buffermatrix_all(int desc, char *data, int datalen,
                                  int nbfrags, struct encode_chunk_context *ctx,
                                  int max) {
  int i;

  for (i = 0; i < max; i++) {
    int err = encode_chunk_buffermatrix(desc, data, datalen, nbfrags, ctx, i, ctx->chunk_size);
    if (err != 0) {
      return err;
    }
  }
  return 0;
}

int my_liberasurecode_encode_buffermatrix_cleanup(int desc, size_t len,
                                                  char **encoded_data,
                                                  char **encoded_parity) {
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
