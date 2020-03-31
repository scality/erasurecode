package erasurecode

import (
	"bytes"
	"io"
)

type BufferMatrix struct {
	b                []byte
	zero             []byte
	hdrSize, bufSize int
	len              int // len of input
	k                int
	curBlock         int
	leftInBlock      int
	finished         bool
}

// FragLen returns the size of a "fragment" aligned to a block size (data + header)
func (b BufferMatrix) FragLen() int {
	return b.SubGroups() * (b.bufSize + b.hdrSize)
}

// SubGroups returns the number of blocks inside a single fragment
func (b BufferMatrix) SubGroups() int {
	nbBlocks := (b.len + b.bufSize - 1) / b.bufSize
	nbStripes := (nbBlocks + b.k - 1) / b.k
	return nbStripes
}

func (b BufferMatrix) maxLen() int {
	return (b.SubGroups() * b.k) * (b.bufSize + b.hdrSize)
}

// NewBufferMatrix returns a new buffer suitable for <len> data and organized
// such as it can be injected into EncodeMatrixWithBuffer without allocation/copying
// the data into shards
func NewBufferMatrix(bufSize int, len int, k int) *BufferMatrix {
	var b BufferMatrix
	b.Reset(bufSize, len, k)
	return &b
}

// Reset serves the same purpose as NewBufferMatrix but use the existing buffer and
// tries to avoid allocation of the underlying buffer.
func (b *BufferMatrix) Reset(bufSize int, length int, k int) {
	hdrSize := fragmentHeaderSize()
	b.hdrSize = hdrSize
	b.bufSize = bufSize
	b.len = length
	b.k = k
	b.leftInBlock = -1
	b.curBlock = 0
	b.finished = false

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
}

var emptyErasureHeader = bytes.Repeat([]byte{0}, fragmentHeaderSize())

// getOffset returns current offset in buffer and size left in the current block
// So that it is safe to copy <left> bytes at <offset>.
// If we are at a boundary, it will init the header and skip it.
func (b *BufferMatrix) getOffset() (int, int) {
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

// Finish *must* be called after the final Write() *before* using the buffer
// in EncodeMatrix
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

func (b BufferMatrix) getRealBlock(blockidx int) int {
	subgroup := b.SubGroups()
	return (blockidx%b.k)*subgroup + (blockidx / b.k)
}

func (b *BufferMatrix) Write(p []byte) (int, error) {
	var dataCopied int

	for len(p) > 0 {
		curOffset, leftToCopy := b.getOffset()

		var max int

		if len(p) > leftToCopy {
			max = leftToCopy
		} else {
			max = len(p)
		}

		n := copy(b.b[curOffset:], p[:max])

		b.leftInBlock -= n
		dataCopied += max
		if b.leftInBlock == 0 {
			b.curBlock++
			b.leftInBlock--
		}

		p = p[max:]
	}
	return dataCopied, nil
}

func (b *BufferMatrix) ReadFrom(r io.Reader) (int64, error) {
	read := int64(0)

	for {
		curOffset, max := b.getOffset()

		n, err := r.Read(b.b[curOffset : curOffset+max])
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
		res = append(res, b.b[curOffset:curOffset+b.bufSize]...)
	}

	return res[:b.len]
}

func (b BufferMatrix) Bytes() []byte {
	return b.b
}

func (b BufferMatrix) Length() int {
	return b.len
}
