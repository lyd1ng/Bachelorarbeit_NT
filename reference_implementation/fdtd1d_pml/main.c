#ifndef fp
#define fp float
#endif

#define UWIDTH 1026

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fdtd_helper.h"

fp old_ey_universe[UWIDTH] = {0};
fp old_hx_universe[UWIDTH] = {0};
fp current_ey_universe[UWIDTH] = {0};
fp current_hx_universe[UWIDTH] = {0};
fp sigmaz_universe[UWIDTH] = {0};
fp permittivity_universe[UWIDTH] = {0};
fp permeability_universe[UWIDTH] = {0};

fp ey_factor1_universe[UWIDTH] = {0};             
fp ey_factor2_universe[UWIDTH] = {0};             
fp hx_factor1_universe[UWIDTH] = {0}; 
fp hx_factor2_universe[UWIDTH] = {0};

fp ramped_sin_frequency;
fp fraction;
int samplerate_space;
int samplerate_time;
fp grid_width_z;
fp time_step;
fp current_time;
fp max_time;

fp ramped_sin_frequency;
fp ramped_sin_amplitude;
int ramped_sin_length;



void print_simulation_parameter()
{
	printf("# dimensions: %d\n", UWIDTH);
	printf("# space_sr: %d\n", samplerate_time);
	printf("# time_sr: %d\n", samplerate_space);
	printf("# floating point: %d\n", 8 * sizeof(fp));
	printf("# ramped_sin_frequency: %f\n", ramped_sin_frequency);
	printf("# ramped_sin_amplitude: %f\n", ramped_sin_amplitude);
	printf("# ramped_sin_length: %d\n", ramped_sin_length);
	printf("# max_time: %.32f\n", max_time);
}



void init_params()
{
	ramped_sin_frequency = 2.4 * pow(10, 9);
	ramped_sin_amplitude = 1;
	ramped_sin_length = 10;

	fraction = 1;
	samplerate_space = 20;
	samplerate_time = 10;
	grid_width_z = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);
	time_step = get_time_resolution_1d(fraction, grid_width_z) / (fp)samplerate_time;
	current_time = 0;
	max_time = 10000 * time_step;

	for (int x=0; x < UWIDTH; x++)
	{
		permittivity_universe[x] = 1;
		permeability_universe[x] = 1;
	}
}


void set_pml_geometry()
{
    int pml_width = 40;
    fp f = vacuum_permittivity / (2.0 * time_step);
    for (int x=0; x < pml_width; x++)
    {
	sigmaz_universe[pml_width - x] = f * pow((fp)x / (fp)pml_width, 3);
        sigmaz_universe[UWIDTH - pml_width + x] = f * pow((fp)x / (fp)pml_width, 3);
    }
}

void calculate_factors()
{
    for (int z=0; z < UWIDTH; z++)
    {
	    ey_factor1_universe[z] = 1 - ((time_step * sigmaz_universe[z]) / vacuum_permittivity);
	    ey_factor2_universe[z] = get_electric_update_coefficient_1d(time_step, permittivity_universe[z]);
	    hx_factor1_universe[z] = (-sigmaz_universe[z] / (2 * vacuum_permittivity) + 1.0 / time_step)
		    / (sigmaz_universe[z] / (2 * vacuum_permittivity) + 1.0 / time_step);
	    hx_factor2_universe[z] = (light_speed / permeability_universe[z]) /
		    (sigmaz_universe[z] / (2 * vacuum_permittivity) + 1.0 / time_step);
    }
}


void handle_perfect_electrical_conductors()
{
	for (int x=0; x < UWIDTH; x++)
	{
		if (permittivity_universe[x] == 12345)
		{
			current_ey_universe[x] = 0.0;
		}
	}
}

void handle_perfect_magnetical_conductors()
{
	for (int x=0; x < UWIDTH; x++)
	{
		if (permeability_universe[x] == 12345)
		{
			current_hx_universe[x] = 0.0;
		}
	}
}

void update_fields()
{
	// Update the magnetic field
	for (int x=1; x < UWIDTH - 1; x++)
	{
		fp hx_curl_term = ((old_ey_universe[x + 1] - old_ey_universe[x]) / grid_width_z);
		current_hx_universe[x] = hx_factor1_universe[x] * old_hx_universe[x] + hx_factor2_universe[x] * hx_curl_term;
	}
	handle_perfect_magnetical_conductors();


	// Update the electrical field
	for (int x=1; x < UWIDTH - 1; x++)
	{
		fp curl_term = ((current_hx_universe[x] - current_hx_universe[x - 1]) / grid_width_z);
            	current_ey_universe[x] = ey_factor1_universe[x] * old_ey_universe[x] + ey_factor2_universe[x] * curl_term;
	}
	handle_perfect_electrical_conductors();

	int middle_x = (int)UWIDTH / 2;
	current_ey_universe[middle_x] = ramped_sinus(current_time, ramped_sin_length * time_step, ramped_sin_amplitude, ramped_sin_frequency);

	memcpy(old_ey_universe, current_ey_universe, sizeof(fp) * UWIDTH);
	memcpy(old_hx_universe, current_hx_universe, sizeof(fp) * UWIDTH);
}

int main()
{
	init_params();
	set_pml_geometry();
	calculate_factors();
	print_simulation_parameter();
	while (current_time < max_time)
	{
		update_fields();
		current_time += time_step;
		
	}

	for (int x = 0; x < UWIDTH; x++)
	{
		printf("%d\t%.32f\n", x, current_ey_universe[x]);
	}
	return 0;
}
