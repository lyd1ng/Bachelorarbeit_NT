#ifndef FDTD_HELPER
#define FDTD_HELPER

#ifndef fp
#define fp float
#endif

const int light_speed;
const fp vacuum_permittivity;
const fp vacuum_permeability;

fp get_lambda_min(fp max_frequency, fp max_fraction);
fp get_grid_width(fp max_frequency, fp max_fraction, fp samplerate);
fp get_time_resolution_2d(fp min_fraction, fp grid_width_x, fp grid_width_y);
fp get_time_resolution_3d(fp min_fraction, fp grid_width_x, fp grid_width_y, fp grid_width_z);
fp ramped_sinus(fp t, fp ramp_len, fp amplitude, fp frequency);

#endif

