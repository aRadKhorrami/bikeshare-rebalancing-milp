Bikeshare Rebalancing: Final MILP Model
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Welcome to the documentation!
This project implements the exact MILP model from the final proposal (November 2025)
using real Capital Bikeshare October 2025 data.

Key Features
------------
- Real 624,869 trips from October 2025
- PySCIPOpt / Gurobi solver
- Interactive Streamlit dashboard
- Fleet size constraint

Mathematical Model
------------------
Full formulation is available in the `proposal PDF <https://github.com/aRadKhorrami/bikeshare-rebalancing-milp/blob/main/Bikeshare_Rebalancing_Final_Proposal.pdf>`_

Sample Input Files
------------------
The project uses real Capital Bikeshare data from October 2025:

    Trip data: https://s3.amazonaws.com/capitalbikeshare-data/index.html
    
    Station locations: https://opendata.dc.gov/datasets/DCGIS::capital-bikeshare-locations/explore

Modules
-------
data
model
app