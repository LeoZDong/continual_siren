# Continual Learning in SIREN

Fit SIREN under a continual learning setup (for images). Specifically, we divide the image into `M` regions. For the first `1/M` steps, we only feed coordinates from the first region to SIREN for supervision. For the next `1/M` steps, we only feed coordinates from the next region and so on.

Modify `yaml` configuration files to change experiment setup. Some notable parameters:

- `data.divide_side_n`: Make `N` divisions on each side. This results in `M = N^2` regions in total.
- `trainer.switch_region_every_steps`: How frequently we switch to the next region. By default, we would "wrap around" to the first region if we switch more times than the total number of regions. But for a more realistic setting, we should switch exactly `M` times.
- `trainer.continual`: Whether we experiment in the continual learning setting. If `False`, we run a non-continual baseline that feeds coordinates randomly from the **full** grid.

Link to [experiment logs](https://docs.google.com/spreadsheets/d/1NMEjlQpJXU-L728Af8WB71ll8Jz5s8GONhvxnAzex28/edit?usp=sharing).
