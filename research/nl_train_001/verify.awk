# Rows of expected.txt: six reference logits, then zero-based reference label.
function abs(x) { return x < 0 ? -x : x }
function fail(message) { print "CHECK_FAIL " message > "/dev/stderr"; failed=1; exit 1 }
BEGIN { rows = 0 }
NR == FNR {
    if (NF != 7 || $7 !~ /^[0-5]$/) fail("invalid reference row " FNR)
    ref[FNR - 1] = $0
    refs++
    next
}
$1 == "DATA_PREDICT" {
    if ($2 != "index=" rows || substr($3,1,7) != "values=" || NF != 3 || rows >= refs)
        fail("unexpected output row " rows)
    if (split(substr($3,8), a, ",") != 6) fail("expected six logits")
    split(ref[rows], b)
    ai = bi = 1
    for (j=1; j<=6; j++) {
        if (a[j] !~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) fail("nonfinite or invalid logit")
        err=abs((a[j]+0) - (b[j]+0))
        if (err > max_err) max_err=err
        if (err > 2e-8) fail("logit mismatch at row " rows)
        if ((a[j]+0) > (a[ai]+0)) ai=j
        if ((b[j]+0) > (b[bi]+0)) bi=j
    }
    if (ai != bi) fail("class mismatch at row " rows)
    if (ai - 1 == b[7]+0) correct++
    rows++
}
END {
    if (failed) exit 1
    if (refs != 2947 || rows != refs) fail("missing rows")
    printf "HAR_PASS rows=%d correct=%d accuracy=%.9f max_abs_logit_error=%.12g\n", rows, correct, correct/rows, max_err
}
