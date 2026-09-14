<!--
SPDX-FileCopyrightText: PyPSA-Earth and PyPSA-Eur Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Rule `add_offgrid_components`

!!! note "Workflow Diagram"
    See the complete workflow in the repository.

The optional `add_offgrid_components` rule inserts isolated offgrid electricity
systems immediately before optimisation when `offgrid.enable` is true. It keeps
the original regional AC electricity load at its on-grid bus, adds one isolated
offgrid bus per final AC bus, and lets the optimiser choose a fixed profile
share of regional electricity demand for isolated local supply.

If access-rate data is provided, existing load is interpreted as served demand
and scaled to total demand via `D_total = D_served / access_rate`. If
`offgrid.base_demand_year` or `offgrid.base_demand_file` is provided, this
baseline profile is treated as already connected on-grid demand. Only demand
above the baseline profile is eligible for the endogenous grid-vs-offgrid split.
In sector-coupled networks the same isolated offgrid demand-share mechanism is
used, but oil backup is represented as an oil-to-electricity link with optional
CO2 output to `co2 atmosphere`.

By default, offgrid PV, battery, and oil/diesel use the same processed costs as
their on-grid counterparts. A scenario may instead set `offgrid.costs` with
explicit CAPEX and diesel marginal cost assumptions. CAPEX values are
annualised before they are written to the network.

Offgrid PV shares the same regional solar potential as on-grid PV. During the
solve step, an additional constraint limits the sum of on-grid and offgrid
solar capacity at each AC bus to the original on-grid solar potential.

## Script Documentation

::: scripts.add_offgrid_components
    options:
        show_root_heading: false
        show_source: false
