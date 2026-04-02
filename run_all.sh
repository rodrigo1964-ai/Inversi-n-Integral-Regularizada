#!/bin/bash
# ============================================================
# run_all.sh — Ejecutar todos los Case Studies del 10Paper
#
# Uso:
#   ./run_all.sh              # Todo (tests + figuras)
#   ./run_all.sh --tests      # Solo tests
#   ./run_all.sh --figures    # Solo figuras
#
# Autor: Rodolfo H. Rodrigo — UNSJ — 2026
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

MODE="${1:-all}"

header() {
    echo ""
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${NC}"
}

run_step() {
    local label="$1"
    local dir="$2"
    local script="$3"

    echo -e "\n${BOLD}▶ ${label}${NC}"
    cd "$SCRIPT_DIR/$dir"
    if python3 "$script"; then
        echo -e "${GREEN}  ✓ ${label} — OK${NC}"
    else
        echo -e "${RED}  ✗ ${label} — FAILED${NC}"
        FAILURES=$((FAILURES + 1))
    fi
}

FAILURES=0
T_START=$SECONDS

# ── Tests ────────────────────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "--tests" ]; then
    header "TESTS"

    run_step "CaseStudy_1 tests (clean data accuracy)"        CaseStudy_1  test_clean.py
    run_step "CaseStudy_2 tests (noise-dominated regime)"     CaseStudy_2  test_noise.py
    run_step "CaseStudy_3 tests (TII performance)"            CaseStudy_3  test_tii.py
    run_step "CaseStudy_4 tests (direct regressor robustness)" CaseStudy_4  test_robustness.py
    run_step "CaseStudy_5 tests (EKF deriv. vs integral)"     CaseStudy_5  test_ekf.py
    run_step "CaseStudy_6 tests (comprehensive comparison)"   CaseStudy_6  test_comparison.py
fi

# ── Figuras ──────────────────────────────────────────────────
if [ "$MODE" = "all" ] || [ "$MODE" = "--figures" ]; then
    header "FIGURAS"

    run_step "CaseStudy_1 figures (Fig 1, Fig 2a-b, Table III)"  CaseStudy_1  generate_figures.py
    run_step "CaseStudy_2 figures (Fig 2c, Table IV)"            CaseStudy_2  generate_figures.py
    run_step "CaseStudy_3 figures (Fig 2d-e, Table V)"           CaseStudy_3  generate_figures.py
    run_step "CaseStudy_4 figures (Table VI)"                    CaseStudy_4  generate_figures.py
    run_step "CaseStudy_5 figures (Fig 2f, Table VII)"           CaseStudy_5  generate_figures.py
    run_step "CaseStudy_6 figures (Fig 3, Table VIII)"           CaseStudy_6  generate_figures.py
fi

# ── Resumen ──────────────────────────────────────────────────
ELAPSED=$(( SECONDS - T_START ))
MIN=$(( ELAPSED / 60 ))
SEC=$(( ELAPSED % 60 ))

header "RESUMEN"
echo ""
if [ $FAILURES -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}✓ Todo OK${NC} — ${MIN}m ${SEC}s"
else
    echo -e "  ${RED}${BOLD}✗ ${FAILURES} fallos${NC} — ${MIN}m ${SEC}s"
fi
echo ""

exit $FAILURES
