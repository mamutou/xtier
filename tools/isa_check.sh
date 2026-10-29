#!/usr/bin/env bash
# The libraries a host without a kernel build's ISA loads must contain no code of it: an inline function both engine
# builds use can end up compiled for the wrong ISA in a shared library (lkqmoe 0.5.3: std::to_string merged into its
# AVX512 copy, SIGILL on Zen2; hence one library per ISA). Checks the linked libraries, not the objects, and that the
# ISA build does contain such code (else the pattern would prove nothing).
#   x86-64:  libxtier_engine.so / _avx2.so / mailbox / tables hold no AVX512 instruction, _avx512.so does
#   aarch64: libxtier_engine.so / _neon.so hold no BFDOT/BFMMLA, _neonbf16.so does
#   tools/isa_check.sh [LIB_DIR=xtier/engine/lib]
set -euo pipefail
dir=${1:-$(cd "$(dirname "$0")/.." && pwd)/xtier/engine/lib}
# zmm / mask registers, the EVEX-only xmm16-31 / ymm16-31, EVEX-only mnemonics on xmm / ymm
x86pattern='zmm|%k[0-7]|%[xy]mm(1[6-9]|2[0-9]|3[01])|^(vpternlog|vperm[it]2|vpcompress|vpexpand|vcompressp|vexpandp|vdpbf16|vcvtne2?ps2bf16|vfixupimm|vrndscale|vgetexp|vgetmant|vscalef|vrcp14|vrsqrt14|vpmullq|vpro[lr][dq]|vpsraq|vpabsq|vpm(ax|in)[su]q|vpermb|vpmultishiftqb|k(mov|and|or|xor|test|shift|not|unpck))'
# Armv8.6 BF16: the BF16 dot product and the BFloat16 matrix multiply-accumulate
armpattern='^(bfdot|bfmmla)'
case "$(uname -m)" in
  aarch64|arm64) pattern=$armpattern; plain='libxtier_engine.so libxtier_engine_neon.so libxtier_mailbox.so libxtier_tables.so'; isa=neonbf16
                 label='non-baseline (BF16) instructions' ;;
  *)             pattern=$x86pattern; plain='libxtier_engine.so libxtier_engine_avx2.so libxtier_mailbox.so libxtier_tables.so'; isa=avx512
                 label='AVX512 instructions' ;;
esac
count() {   # matching instructions in $1 (instruction text only: the symbol names in <...> are dropped)
  objdump -d --no-show-raw-insn "$1" | awk -F'\t' 'NF >= 2 {print $2}' | sed 's/<[^>]*>//g' | grep -cE "$pattern" || true
}
fail=0
for lib in $plain; do
  if [ ! -f "$dir/$lib" ]; then echo "$lib: not built"; continue; fi
  n=$(count "$dir/$lib")
  echo "$lib: $n $label"
  [ "$n" = 0 ] || fail=1
done
n=$(count "$dir/libxtier_engine_$isa.so")
echo "libxtier_engine_$isa.so: $n $label (must be > 0)"
[ "$n" -gt 0 ] || fail=1
[ "$fail" = 0 ] && echo "ISA-CHECK PASS" || echo "ISA-CHECK FAIL"
exit $fail
