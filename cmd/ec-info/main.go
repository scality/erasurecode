package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/benoit-a/erasurecode"
)

func init() {
	flag.Usage = func() {
		fmt.Println("usage: ec-info file1 [... fileN]")
		fmt.Println()
		fmt.Println("Print information about the fragment archives, such as:")
		fmt.Println("  - erasure coding backend used")
		fmt.Println("  - index of the archive")
		fmt.Println("  - number of fragments in the archive")
		fmt.Println("  - number of bytes per fragment")
		fmt.Println("  - errors detected in the archive")
	}
}

func handleFile(fname string) {
	fd, err := os.Open(fname)
	if err != nil {
		fmt.Printf("Error opening %q: %v\n", fname, err)
		return
	}
	defer fd.Close()

	fmt.Printf("Inspecting %q:\n", fname)
	found, offset := 0, 0
	var baseline erasurecode.FragmentInfo
	var origDataSize uint64
	gotEOF := false
	sizeCheckMessage := ""
	for {
		frag, err := erasurecode.ReadFragment(fd)
		if err == io.EOF {
			gotEOF = true
			break
		}
		if sizeCheckMessage != "" {
			// We wait until now to report so we don't flag the last fragment
			fmt.Print(sizeCheckMessage)
		}
		found++
		if err != nil {
			fmt.Printf("    Error reading frag %v (offset 0x%08x): %v\n", found, offset, err)
			offset += len(frag) // Keep our byte count up-to-date
			break
		}

		info := erasurecode.GetFragmentInfo(frag)
		if baseline.Size == 0 {
			baseline = info
			fmt.Printf("    Index: %2v FragSize: %5v ", info.Index, info.Size)
			fmt.Printf("Backend: %v/%v ", info.BackendName, info.BackendVersion)
			fmt.Printf("libec/%v\n", info.ErasureCodeVersion)
		}
		origDataSize += info.OrigDataSize

		if info.BackendName != baseline.BackendName {
			fmt.Printf("    Fragment %v (offset 0x%08x) has unexpected backend %v\n", found, offset, info.BackendName)
		}
		if info.Index != baseline.Index {
			fmt.Printf("    Fragment %v (offset 0x%08x) has unexpected index %v\n", found, offset, info.Index)
		}
		if info.Size != baseline.Size {
			sizeCheckMessage = fmt.Sprintf("    Fragment %v (offset 0x%08x) has unexpected size %v\n", found, offset, info.Size)
		} else {
			sizeCheckMessage = ""
		}
		offset += len(frag)
	}

	if gotEOF {
		fmt.Printf("    Found %v fragments, totaling %v bytes (original file was %v bytes)\n\n", found, offset, origDataSize)
	} else {
		fmt.Printf("    Found %v fragments, totaling %v bytes before aborting\n\n", found, offset)
	}
}

func main() {
	flag.Parse()
	files := flag.Args()
	if len(files) == 0 {
		flag.Usage()
		fmt.Printf("\nexpected at least one file to inspect")
		os.Exit(1)
	}

	for _, fname := range files {
		handleFile(fname)
	}
}
