# OpenSUSE (Leap_16.X/Tumbleweed_2026/Slowroll_2026)
# Note: Compiles only the "ollama" Executable
#		Tested on OpenSUSE Linux 64-bit (x86_64)
#
sudo chmod -R 777 ~/Builds/ollama-SRC-0.30.0-rc29
cd ~/Builds/ollama-SRC-0.30.0-rc29
tar -C go-new -xzf go1.24.13.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/cuda/bin
export PATH=~/Builds/ollama-SRC-0.30.0-rc29/go-new/go/bin:$PATH
export MAKEFLAGS="-j4 -fno-aggressive-loop-optimizations"
export GOMAXPROCS=4
go clean -cache
go generate ./...
go build -o ollama .

---- or ----

cd ~/Builds/ollama-SRC-0.30.0-rc29/; export PATH=$PATH:/usr/local/cuda/bin; export PATH=~/Builds/go-1.24.13-toolchain/go/bin/:$PATH; export MAKEFLAGS="-j2 -fno-aggressive-loop-optimizations"; export GOMAXPROCS=2; go clean -cache; go generate ./...; go build -o ollama-cjz_0_30_0_xc29-linux-amd64 .