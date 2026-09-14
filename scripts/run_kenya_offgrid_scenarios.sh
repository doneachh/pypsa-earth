#!/usr/bin/env bash
# SPDX-FileCopyrightText: PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

cores="${CORES:-4}"
snakemake_bin="${SNAKEMAKE:-snakemake}"
base_config="config.tutorial.yaml"
scenario_dir="configs/scenarios"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/pypsa-earth-snakemake-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pypsa-earth-matplotlib-cache}"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

run_case() {
    local target="$1"
    local scenario_config="$2"
    shift 2

    echo
    echo ">>> ${target} :: ${scenario_config}"
    "${snakemake_bin}" -j "${cores}" "${target}" --configfile "${base_config}" "${scenario_dir}/${scenario_config}" "$@"
}

extra_args=("$@")

run_case solve_all_networks "config.KE-reference-overnight-2030.yaml" "${extra_args[@]}"
run_case solve_sector_networks "config.KE-reference-overnight-2030.yaml" "${extra_args[@]}"
run_case solve_all_networks "config.KE-offgrid-overnight-2030.yaml" "${extra_args[@]}"
run_case solve_sector_networks "config.KE-offgrid-overnight-2030.yaml" "${extra_args[@]}"

run_case solve_all_networks "config.KE-reference-overnight-2040.yaml" "${extra_args[@]}"
run_case solve_sector_networks "config.KE-reference-overnight-2040.yaml" "${extra_args[@]}"
run_case solve_all_networks "config.KE-offgrid-overnight-2040.yaml" "${extra_args[@]}"
run_case solve_sector_networks "config.KE-offgrid-overnight-2040.yaml" "${extra_args[@]}"

run_case solve_sector_networks_myopic "config.KE-reference-myopic.yaml" "${extra_args[@]}"
run_case solve_sector_networks_myopic "config.KE-offgrid-myopic.yaml" "${extra_args[@]}"
