# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

.PHONY: test setup clean run-kenya-offgrid-smoke dry-run-kenya-offgrid-smoke

test:
	set -e
	python -m pytest test/test_add_offgrid_components.py -q
	# this runs test scenario in test/config.test.yaml
	snakemake solve_all_networks -call --configfile config.tutorial.yaml test/config.test.yaml
	snakemake -n -j1 solve_all_networks --configfile config.tutorial.yaml test/config.test.yaml test/config.offgrid.yaml
	# add custom config to tutorial config
	snakemake solve_all_networks_monte -call --configfile config.tutorial.yaml test/config.monte_carlo.yaml
	snakemake -c4 solve_sector_networks --configfile config.tutorial.yaml test/config.sector.yaml
	snakemake -n -j1 solve_sector_networks --configfile config.tutorial.yaml test/config.sector.yaml test/config.offgrid.yaml
	snakemake -c4 solve_sector_networks_myopic --configfile config.tutorial.yaml test/config.myopic.yaml
	snakemake solve_all_networks -call --configfile config.tutorial.yaml test/config.custom.yaml
	snakemake solve_all_networks -call --configfile config.tutorial.yaml test/config.landlock.yaml
	echo "All tests completed successfully."

setup:
	# Add setup commands here
	echo "Setup complete."

clean:
	# Add clean-up commands here
	snakemake -j1 solve_all_networks --delete-all-output --configfile config.tutorial.yaml test/config.custom.yaml
	snakemake -j1 solve_all_networks --delete-all-output --configfile config.tutorial.yaml configs/scenarios/config.NG.yaml
	snakemake -j1 solve_all_networks_monte --delete-all-output --configfile test/config.monte_carlo.yaml
	snakemake -j1 run_all_scenarios --delete-all-output --configfile test/config.landlock.yaml
	snakemake -j1 solve_sector_networks --delete-all-output --configfile test/config.sector.yaml
	snakemake -j1 solve_sector_networks_myopic --delete-all-output --configfile config.tutorial.yaml test/config.myopic.yaml
	echo "Clean-up complete."

run-kenya-offgrid-smoke:
	bash scripts/run_kenya_offgrid_scenarios.sh

dry-run-kenya-offgrid-smoke:
	bash scripts/run_kenya_offgrid_scenarios.sh -n
