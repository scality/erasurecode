#ifndef BACKEND_H
#define BACKEND_H

#include <liberasurecode/erasurecode.h>
#include <liberasurecode/erasurecode_helpers_ext.h>
#include <liberasurecode/erasurecode_postprocessing.h>


struct encode_chunk_context {
	ec_backend_t instance;           // backend instance
	char **datas;                    // the K datas
	char **codings;                  // the M codings
	unsigned int number_of_subgroup; // number of subchunk in each K part
	unsigned int chunk_size;         // datasize of each subchunk
	unsigned int frags_len;          // allocating size of each K+M objects
	int blocksize;                   // k-bounds of data
	int k;
	int m;
  };

char **makeStrArray(int n);

void freeStrArray(char **arr);

uint64_t getOrigDataSize(struct fragment_header_s *header);

uint32_t getBackendVersion(struct fragment_header_s *header);

ec_backend_id_t getBackendID(struct fragment_header_s *header);

uint32_t getECVersion(struct fragment_header_s *header);

int getHeaderSize(void);

char *linearize(int k, char **in, int inlen, char *dest, uint64_t destlen,
                uint64_t *outlen);

bool check_matrix_fragment(char *frag, int frag_len, int piecesize);

void encode_chunk_prepare(int desc, char *data, int datalen, int piecesize,
                          struct encode_chunk_context *ctx);

size_t get_fragment_header_size(void);

int encode_chunk(int desc, char *data, int datalen,
                 struct encode_chunk_context *ctx, int nth);

int encode_chunk_all(int desc, char *data, int datalen,
                     struct encode_chunk_context *ctx, int max);

int my_liberasurecode_encode_cleanup(int desc, size_t len, char **encoded_data,
                                     char **encoded_parity);
									 
void encode_chunk_buffermatrix_prepare(int desc, char *data, int datalen,
                                       int piecesize, int frags_len,
                                       int number_of_subgroup,
                                       struct encode_chunk_context *ctx);

int encode_chunk_buffermatrix_all(int desc, char *data, int datalen,
                                  int nbfrags, struct encode_chunk_context *ctx,
                                  int max);

int encode_chunk_buffermatrix(int desc, char *data, int datalen, int nbFrags,
                              struct encode_chunk_context *ctx, int nth, size_t frag_len);

int my_liberasurecode_encode_buffermatrix_cleanup(int desc, size_t len,
                                                  char **encoded_data,
                                                  char **encoded_parity);

#endif