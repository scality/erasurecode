DEPDIR=$(PWD)/deps
BUILDDIR=$(PWD)/build

LIBERASURE=$(DEPDIR)/liberasurecode
ISALSRC=$(DEPDIR)/isa-l
LINT_START="983e1d15"

.PHONY: default test clean pretty cmds

default: $(BUILDDIR)/lib/liberasurecode.a $(BUILDDIR)/lib/libisal.a cmds
	PKG_CONFIG_PATH=$(BUILDDIR)/lib/pkgconfig \
	go build

test: $(BUILDDIR)/lib/liberasurecode.a $(BUILDDIR)/lib/libisal.a
	DYLIB_LIBRARY_PATH=$(BUILDDIR)/lib \
	LD_LIBRARY_PATH=$(BUILDDIR)/lib \
	PKG_CONFIG_PATH=$(BUILDDIR)/lib/pkgconfig \
	go test -v .

lint:
	golangci-lint run --new-from-rev ${LINT_START}

cmds: ec-split ec-info

ec-split: $(PWD)/cmd/ec-split/main.go $(PWD)/backend.go $(PWD)/streaming.go
	PKG_CONFIG_PATH=$(BUILDDIR)/lib/pkgconfig \
	go build github.com/scality/erasurecode/cmd/ec-split

ec-info: $(PWD)/cmd/ec-info/main.go $(PWD)/backend.go $(PWD)/streaming.go
	PKG_CONFIG_PATH=$(BUILDDIR)/lib/pkgconfig \
	go build github.com/scality/erasurecode/cmd/ec-info

clean:
	rm -rf $(BUILDDIR) $(DEPDIR)

pretty:
	find $(PWD) -name '*.go' | xargs gofmt -l -w

$(ISALSRC)/autogen.sh:
	git clone --depth 1 --branch v2.28.0 https://github.com/intel/isa-l.git $(ISALSRC)

$(LIBERASURE)/autogen.sh:
	git clone --depth 1 --branch v1.6.1 https://github.com/scality/liberasurecode.git $(LIBERASURE)

$(PWD)/deps/%/configure: $(PWD)/deps/%/autogen.sh
	cd $(@D) && ./autogen.sh

$(PWD)/deps/%/Makefile: $(PWD)/deps/%/configure
	cd $(@D) && ./configure --prefix=$(BUILDDIR)

$(BUILDDIR)/lib/liberasurecode.a: $(LIBERASURE)/Makefile
	cd $(LIBERASURE) && make && make install && sudo ldconfig

$(BUILDDIR)/lib/libisal.a: $(ISALSRC)/Makefile
	cd $(ISALSRC) && make && make install && sudo ldconfig
