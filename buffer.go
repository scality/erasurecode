package erasurecode

import (
	"bytes"
	"io"
)

type BufferInfo struct {
	hdrSize, bufSize   int
	len                int
	k                  int
	curBlock           int
	leftInBlock        int
	sizeOfLastSubGroup int
	newStyle           bool
}

// FragLen returns the size of a "fragment" aligned to a block size (data + header)
func (b BufferInfo) FragLen() int {
	return b.SubGroups() * (b.bufSize + b.hdrSize)
}

// SubGroups returns the number of blocks inside a single fragment
func (b BufferInfo) SubGroups() int {
	nbBlocks := (b.len + b.bufSize - 1) / b.bufSize
	nbStripes := (nbBlocks + b.k - 1) / b.k
	return nbStripes
}

func (b BufferInfo) maxLen() int {
	return (b.SubGroups() * b.k) * (b.bufSize + b.hdrSize)
}

func (b BufferInfo) IsBlockInLastSubGroup(block int) bool {
	cur := block / b.k
	return cur == b.SubGroups()-1
}

func (b BufferInfo) ComputeSizeOfLastSubGroup() int {
	// total of size already in previous subgroups
	lastSubGroup := b.SubGroups() - 1
	totalSizeInPreviousSubGroups := lastSubGroup * b.k * (b.bufSize)
	leftSize := b.len - totalSizeInPreviousSubGroups
	return leftSize
}

func (b BufferInfo) FragLenLastSubGroup() int {
	if !b.newStyle {
		return b.bufSize
	}
	r := b.ComputeSizeOfLastSubGroup() / b.k
	if b.ComputeSizeOfLastSubGroup()%b.k != 0 {
		r++
	}
	return r
}

func (b *BufferInfo) init(bufSize int, length int, k int) {
	b.newStyle = true

	hdrSize := fragmentHeaderSize()
	b.hdrSize = hdrSize
	b.bufSize = bufSize
	b.len = length
	b.k = k
	b.leftInBlock = -1
	b.curBlock = 0

	b.sizeOfLastSubGroup = b.FragLenLastSubGroup()
}

func NewBufferInfo(bufSize int, length int, k int) *BufferInfo {
	var b BufferInfo
	b.init(bufSize, length, k)
	return &b
}

type BufferMatrix struct {
	b        []byte
	zero     []byte
	finished bool
	BufferInfo
}

// NewBufferMatrix returns a new buffer suitable for <len> data and organized
// such as it can be injected into EncodeMatrixWithBuffer without allocation/copying
// the data into shards
func NewBufferMatrix(bufSize int, length int, k int) *BufferMatrix {
	var b BufferMatrix
	b.Reset(bufSize, length, k)
	return &b
}

// Reset serves the same purpose as NewBufferMatrix but use the existing buffer and
// tries to avoid allocation of the underlying buffer.
func (b *BufferMatrix) Reset(bufSize int, length int, k int) {
	b.init(bufSize, length, k)

	maxLen := b.maxLen()

	if cap(b.b) < maxLen {
		// No internal buffer or buffer is too small, let's replace it
		b.b, _ = memalign(maxLen, defaultAlign)
	}

	// Set proper size of buffer
	b.b = b.b[:maxLen]

	if len(b.zero) < bufSize {
		b.zero = make([]byte, bufSize)
	}
	b.finished = false
}

// UseNewFormat sets the buffer to use the new format.
// The new format is more efficient for the last stripe/subgroup.
// Note: will panic if called after any Write() or ReadFrom()
func (b *BufferMatrix) UseNewFormat() {
	if b.curBlock != 0 || b.leftInBlock != -1 || b.finished {
		panic("UseNewOffset must be called before any Write")
	}
	b.newStyle = true
}

func (b *BufferMatrix) UseOldFormat() {
	if b.curBlock != 0 || b.leftInBlock != -1 || b.finished {
		panic("UseNewOffset must be called before any Write")
	}
	b.newStyle = false
}

// getOffset is a wrapper around getOffsetOld and getOffsetNew.
// It will call the right one depending on the newStyle flag.
func (b *BufferMatrix) getOffset() (int, int) {
	if b.newStyle {
		return b.getOffsetNew()
	}
	return b.getOffsetOld()
}

var emptyErasureHeader = bytes.Repeat([]byte{0}, fragmentHeaderSize())

// Finish *must* be called after the final Write() *before* using the buffer
// It is safe to call it multiple times.
func (b *BufferMatrix) Finish() {
	if b.finished {
		return
	}
	maxBlock := b.SubGroups() * b.k

	for b.curBlock < maxBlock {
		curOffset, leftToCopy := b.getOffset()

		n := copy(b.b[curOffset:], b.zero[0:leftToCopy])

		b.leftInBlock -= n
		if b.leftInBlock == 0 {
			b.curBlock++
			b.leftInBlock--
		}
	}
	b.finished = true
}

// In b.b buffer, the data is organized as follow:
// - for each block, we have a header of size hdrSize
// - then the data of size bufSize
// - then the next block
// - etc.
// The data is organized in stripes of k blocks.
// It is meant to be split later and stored as shards.
// Shard 0 will contain block 0, k, 2k, 3k, etc.
// Shard 1 will contain block 1, k+1, 2k+1, 3k+1, etc.
// etc.
// So b.b is organized as follow:
// [hdr][block 0][hdr][block k][hdr][block 2k][hdr][block 3k] ... [hdr][block 1][hdr][block k+1] etc...
// When writing to buffer, we will write in the current block until it is full.
// Then we will skip the header and write in the next block.
// For example when block 0 is full, we will skip the header and write in block 1. When block 1 is full, we will skip the header and write in block 2, etc.
// Them, when all blocks from 0 to k-1 are full, we will write in block k, k+1, etc.

// getRealBlock returns the real block index in the buffer.
// For example, if we have k=2 and 5 blocks in total, the buffer will be organized as follow:
// [hdr][block 0]   [hdr][block 2]   [hdr][block 4]  [hdr][block 1]  [hdr][block 3]
// So getRealBlock(0) will return 0, getRealBlock(1) will return 3, getRealBlock(2) will return 1, getRealBlock(3) will return 4, getRealBlock(4) will return 2.

// getRealBlock returns the real block index in the buffer.
// blockidx is the block index in the incoming data (0-indexed)
// the return value is the block index in the buffer (0-indexed)
func (b BufferMatrix) getRealBlock(blockidx int) int {
	nbStripes := b.SubGroups()
	return (blockidx%b.k)*nbStripes + (blockidx / b.k)
}

// getOffSetNew returns current offset in buffer and size left in the current block
// Same a getOffset when blocks are not in the last stripe/subgroup
// When blocks are in the last stripe/subgroup, it will split the size left in k parts
// and return the offset and size left for the current block.
// For example, if we have 5*blocksize bytes and k=2, the buffer will be organized as follow:
// [hdr][block 0]  [hdr][block 2]  [hdr][block 4] / [hdr][block 1]  [hdr][block 3]  [hdr][block 5]
// where size(block4) + size(block5) == len - (4 * blocksize) == size of last subgroup
// when/if the size of last subgroup is not divisible by k, the block 4 may be one byte longer than block 5
func (b *BufferMatrix) getOffsetNew() (int, int) {
	realCurBlock := b.getRealBlock(b.curBlock)
	blockSize := b.hdrSize + b.bufSize
	blockOffset := realCurBlock * blockSize
	if b.leftInBlock == -1 {
		// Start of a block
		copy(b.b[blockOffset:], emptyErasureHeader)
		if b.IsBlockInLastSubGroup(b.curBlock) {
			b.leftInBlock = b.FragLenLastSubGroup()
		} else {
			b.leftInBlock = b.bufSize
		}
	}

	bufSize := b.bufSize
	if b.IsBlockInLastSubGroup(b.curBlock) {
		bufSize = b.FragLenLastSubGroup()
	}

	curOffset := blockOffset + (bufSize - b.leftInBlock) + b.hdrSize

	return curOffset, b.leftInBlock
}

// getOffset returns current offset in buffer and size left in the current block
// So that it is safe to copy <left> bytes at <offset>.
// If we are at a boundary, it will init the header and skip it.
func (b *BufferMatrix) getOffsetOld() (int, int) {
	realCurBlock := b.getRealBlock(b.curBlock)
	blockSize := b.hdrSize + b.bufSize
	blockOffset := realCurBlock * blockSize
	if b.leftInBlock == -1 {
		// Start of a block
		copy(b.b[blockOffset:], emptyErasureHeader)
		b.leftInBlock = b.bufSize
	}

	curOffset := blockOffset + (b.bufSize - b.leftInBlock) + b.hdrSize

	return curOffset, b.leftInBlock
}

func (b *BufferMatrix) Write(p []byte) (int, error) {
	var dataCopied int

	for len(p) > 0 {
		curOffset, leftToCopy := b.getOffset()

		m := min(len(p), leftToCopy)

		n := copy(b.b[curOffset:], p[:m])

		b.leftInBlock -= n
		dataCopied += m
		if b.leftInBlock == 0 {
			b.curBlock++
			b.leftInBlock--
		}

		p = p[m:]
	}
	return dataCopied, nil
}

func (b *BufferMatrix) ReadFrom(r io.Reader) (int64, error) {
	read := int64(0)

	for {
		curOffset, m := b.getOffset()

		n, err := r.Read(b.b[curOffset : curOffset+m])
		if err != nil && err != io.EOF {
			return 0, err
		}

		b.leftInBlock -= n
		read += int64(n)

		if b.leftInBlock == 0 {
			b.curBlock++
			b.leftInBlock--
		}
		if err != nil && err == io.EOF {
			break
		}
	}
	b.Finish() // Q: Mark buffer not usable anymore ?
	return read, nil
}

func (b BufferMatrix) RealData() []byte {
	res := make([]byte, 0, b.len)

	for block := 0; len(res) < b.len; block++ {
		blockSize := b.hdrSize + b.bufSize
		curOffset := b.getRealBlock(block)*blockSize + b.hdrSize
		if b.newStyle && b.IsBlockInLastSubGroup(block) {
			amountToCopy := min(b.FragLenLastSubGroup(), b.len-len(res))
			res = append(res, b.b[curOffset:curOffset+amountToCopy]...)
		} else {
			res = append(res, b.b[curOffset:curOffset+b.bufSize]...)
		}
	}

	return res[:b.len]
}

func (b BufferMatrix) Bytes() []byte {
	return b.b
}

func (b BufferMatrix) Length() int {
	return b.len
}

func (b *BufferMatrix) IsBlockInLastSubGroup(block int) bool {
	cur := block / b.k
	return cur == b.SubGroups()-1
}

func (b *BufferMatrix) ComputeSizeOfLastSubGroup() int {
	// total of size already in previous subgroups
	lastSubGroup := b.SubGroups() - 1
	totalSizeInPreviousSubGroups := lastSubGroup * b.k * (b.bufSize)
	leftSize := b.len - totalSizeInPreviousSubGroups
	return leftSize
}

func (b *BufferMatrix) FragLenLastSubGroup() int {
	if !b.newStyle {
		return b.bufSize
	}

	r := b.ComputeSizeOfLastSubGroup() / b.k
	if b.ComputeSizeOfLastSubGroup()%b.k != 0 {
		r++
	}
	return r
}
