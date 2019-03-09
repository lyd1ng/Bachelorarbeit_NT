#include <math.h>

#ifndef fp
#define fp float
#endif

const int light_speed = 299792458;
const fp vacuum_permittivity = (8.854187817 * pow(10, -12));
const fp vacuum_permeability = (1.2566370614 * pow(10, -6));


fp get_lambda_min(fp max_frequency, fp max_fraction)
{
	return light_speed / (max_frequency * max_fraction);
}

fp get_grid_width(fp max_frequency, fp max_fraction, fp samplerate)
{
	return get_lambda_min(max_frequency, max_fraction) / samplerate;
}

fp get_time_resolution_1d(fp min_fraction, fp grid_width_z)
{
	return (min_fraction * grid_width_z) / light_speed;
}

fp get_time_resolution_2d(fp min_fraction, fp grid_width_x, fp grid_width_y)
{
	fp denominator = light_speed * sqrt(pow(grid_width_x, -2) + pow(grid_width_y, -2));
	return min_fraction / denominator;
}

fp get_electric_update_coefficient_1d(fp time_step, fp permittiviy)
{
	return (light_speed * time_step) / permittiviy;
}

fp get_magnetic_update_coefficient_1d(fp time_step, fp permeability)
{
	return (light_speed * time_step) / permeability;
}

fp ramped_sinus(fp t, fp ramp_len, fp amplitude, fp frequency)
{
	fp ramp_factor = 1.0;
	if (t <= ramp_len) { ramp_factor = t / ramp_len; } 
	return ramp_factor * amplitude * sin(t * 2.0 * M_PI * frequency);
}

