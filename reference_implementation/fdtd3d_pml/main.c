#ifndef fp
#define fp float
#endif

#define UWIDTH 64
#define UHEIGHT 64
#define UDEPTH 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fdtd_helper.h"

fp old_ex_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp old_ey_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp old_ez_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp old_hx_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp old_hy_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp old_hz_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};

fp current_ex_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp current_ey_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp current_ez_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp current_hx_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp current_hy_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp current_hz_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};

fp ex_curl_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0}; 
fp ey_curl_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp ez_curl_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp ex_field_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp ey_field_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp ez_field_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hx_curl_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0}; 
fp hy_curl_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hz_curl_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hx_field_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0}; 
fp hy_field_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hz_field_integral_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};

fp sigmax_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp sigmay_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp sigmaz_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};

fp permittivity_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp permeability_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};

fp ex_factor1_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};             
fp ex_factor2_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};             
fp ex_factor3_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp ex_factor4_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp ey_factor1_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};             
fp ey_factor2_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};             
fp ey_factor3_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};             
fp ey_factor4_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};             
fp ez_factor1_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};             
fp ez_factor2_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};             
fp ez_factor3_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp ez_factor4_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hx_factor1_universe[UWIDTH][UHEIGHT][UDEPTH] = {0}; 
fp hx_factor2_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hx_factor3_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hx_factor4_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hy_factor1_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hy_factor2_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hy_factor3_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hy_factor4_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hz_factor1_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hz_factor2_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hz_factor3_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};
fp hz_factor4_universe[UWIDTH][UHEIGHT][UDEPTH] = {0};

fp fraction;
int samplerate_space;
int samplerate_time;
fp grid_width_x;
fp grid_width_y;
fp grid_width_z;
fp time_step;
fp current_time;
fp max_time;

fp ramped_sin_frequency;
fp ramped_sin_amplitude;
int ramped_sin_length;



void print_simulation_parameter()
{
	printf("# dimensions: %dx%dx%d\n", UWIDTH, UHEIGHT, UDEPTH);
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
	grid_width_x = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);
	grid_width_y = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);
	grid_width_z = get_grid_width(ramped_sin_frequency, fraction, samplerate_space);
	time_step = get_time_resolution_3d(fraction, grid_width_x, grid_width_y, grid_width_z) / (fp)samplerate_time;
	current_time = 0;
	max_time = 100 * time_step;

	for (int z=0; z < UDEPTH; z++)
	{
		for (int y=0; y < UHEIGHT; y++)
		{
			for (int x=0; x < UWIDTH; x++)
			{
				permittivity_universe[x][y][z] = 1;
				permeability_universe[x][y][z] = 1;
			}
		}
	}
}


void set_pml_geometry()
{
    int pml_width = 20;
    int pml_height = 20;
	int pml_depth = 20;
    fp f = vacuum_permittivity / (2.0 * time_step);
    for (int z=1; z < UDEPTH - 1; z++)
    {
    	for (int y=1; y < UHEIGHT - 1; y++)
    	{
    	    for (int x=1; x < pml_width; x++)
    	    {
    	        sigmax_universe[pml_width - x - 1][y][z] = f * pow((fp)x / (fp)pml_width, 3);
    	        sigmax_universe[UWIDTH - pml_width + x][y][z] = f * pow((fp)x / (fp)pml_width, 3);
    	    }
    	}
	}
	for (int z=1; z < UDEPTH - 1; z++)
    {
    	for (int x=1; x < UWIDTH - 1; x++)
    	{
    	    for (int y=1; y < pml_height; y++)
    	    {
    	        sigmay_universe[x][pml_height - y - 1][z] = f * pow((fp)y / (fp)pml_height, 3);
    	        sigmay_universe[x][UHEIGHT - pml_height + y][z] = f * pow((fp)y / (fp)pml_height, 3);
    	    }
    	}
    }
	for (int x=1; x < UWIDTH - 1; x++)
	{
		for (int y=1; y < UHEIGHT - 1; y++)
		{
			for (int z=1; z < pml_depth; z++)
			{
				sigmaz_universe[x][y][pml_depth - z - 1] = f * pow((fp)z / (fp)pml_depth, 3);
				sigmaz_universe[x][y][UDEPTH - pml_depth + z] = f * pow((fp)z / (fp)pml_depth, 3);
			}
		}
	}
}

void calculate_factors()
{
	fp hf = 0;
	for (int z=0; z < UDEPTH; z++)
	{
    	for (int y=0; y < UHEIGHT; y++)
    	{
    	    for (int x=0; x < UWIDTH; x++)
			{
				// Calculate the ex factors
    	    	ex_factor1_universe[x][y][z] = 1 - ((time_step * (sigmay_universe[x][y][z] + sigmaz_universe[x][y][z])) / vacuum_permittivity);
    	        ex_factor2_universe[x][y][z] = (sigmay_universe[x][y][z] * sigmaz_universe[x][y][z] * pow(time_step, 2)) / pow(vacuum_permittivity, 2);
    	        ex_factor3_universe[x][y][z] = (light_speed * time_step) / permittivity_universe[x][y][z];
				ex_factor4_universe[x][y][z] = (light_speed * sigmax_universe[x][y][z] * pow(time_step, 2)) / (permittivity_universe[x][y][z] * vacuum_permittivity);

				// Calculate the ey factors
    	    	ey_factor1_universe[x][y][z] = 1 - ((time_step * (sigmax_universe[x][y][z] + sigmaz_universe[x][y][z])) / vacuum_permittivity);
    	        ey_factor2_universe[x][y][z] = (sigmax_universe[x][y][z] * sigmaz_universe[x][y][z] * pow(time_step, 2)) / pow(vacuum_permittivity, 2);
    	        ey_factor3_universe[x][y][z] = (light_speed * time_step) / permittivity_universe[x][y][z];
				ey_factor4_universe[x][y][z] = (light_speed * sigmay_universe[x][y][z] * pow(time_step, 2)) / (permittivity_universe[x][y][z] * vacuum_permittivity);

				// Calculate the ez factors
    	    	ez_factor1_universe[x][y][z] = 1 - ((time_step * (sigmax_universe[x][y][z] + sigmay_universe[x][y][z])) / vacuum_permittivity);
    	        ez_factor2_universe[x][y][z] = (sigmax_universe[x][y][z] * sigmay_universe[x][y][z] * pow(time_step, 2)) / pow(vacuum_permittivity, 2);
    	        ez_factor3_universe[x][y][z] = (light_speed * time_step) / permittivity_universe[x][y][z];
				ez_factor4_universe[x][y][z] = (light_speed * sigmaz_universe[x][y][z] * pow(time_step, 2)) / (permittivity_universe[x][y][z] * vacuum_permittivity);

				// Calculate the hx factors
				hf = (sigmay_universe[x][y][z] + sigmaz_universe[x][y][z]) / (2 * vacuum_permittivity)
					 + (sigmay_universe[x][y][z] * sigmaz_universe[x][y][z] * time_step) / (4 * pow(vacuum_permittivity, 2))
					 + 1 / time_step;
    	        hx_factor1_universe[x][y][z] = (- (sigmay_universe[x][y][z] + sigmaz_universe[x][y][z]) / (2 * vacuum_permittivity)
											    - (sigmay_universe[x][y][z] * sigmaz_universe[x][y][z] * time_step) / (4 * pow(vacuum_permittivity, 2))
											    + 1 / time_step) / hf;
    	        hx_factor2_universe[x][y][z] = (light_speed / permittivity_universe[x][y][z]) /  hf;
    	        hx_factor3_universe[x][y][z] = ((light_speed * sigmax_universe[x][y][z] * time_step) / (permeability_universe[x][y][z] * vacuum_permittivity)) / hf;
				hx_factor4_universe[x][y][z] = ((sigmay_universe[x][y][z] * sigmaz_universe[x][y][z] * time_step) / pow(vacuum_permittivity, 2)) / hf;

				// Calculate the hy factors
				hf = (sigmax_universe[x][y][z] + sigmaz_universe[x][y][z]) / (2 * vacuum_permittivity)
					 + (sigmax_universe[x][y][z] * sigmaz_universe[x][y][z] * time_step) / (4 * pow(vacuum_permittivity, 2))
					 + 1 / time_step;
    	        hy_factor1_universe[x][y][z] = (- (sigmax_universe[x][y][z] + sigmaz_universe[x][y][z]) / (2 * vacuum_permittivity)
											    - (sigmax_universe[x][y][z] * sigmaz_universe[x][y][z] * time_step) / (4 * pow(vacuum_permittivity, 2))
											    + 1 / time_step) / hf;
    	        hy_factor2_universe[x][y][z] = (light_speed / permittivity_universe[x][y][z]) /  hf;
    	        hy_factor3_universe[x][y][z] = ((light_speed * sigmay_universe[x][y][z] * time_step) / (permeability_universe[x][y][z] * vacuum_permittivity)) / hf;
				hy_factor4_universe[x][y][z] = ((sigmax_universe[x][y][z] * sigmaz_universe[x][y][z] * time_step) / pow(vacuum_permittivity, 2)) / hf;

				// Calculate the hz factors
				hf = (sigmax_universe[x][y][z] + sigmay_universe[x][y][z]) / (2 * vacuum_permittivity)
					 + (sigmax_universe[x][y][z] * sigmay_universe[x][y][z] * time_step) / (4 * pow(vacuum_permittivity, 2))
					 + 1 / time_step;
    	        hz_factor1_universe[x][y][z] = (- (sigmax_universe[x][y][z] + sigmay_universe[x][y][z]) / (2 * vacuum_permittivity)
											    - (sigmax_universe[x][y][z] * sigmay_universe[x][y][z] * time_step) / (4 * pow(vacuum_permittivity, 2))
											    + 1 / time_step) / hf;
    	        hz_factor2_universe[x][y][z] = (light_speed / permittivity_universe[x][y][z]) /  hf;
    	        hz_factor3_universe[x][y][z] = ((light_speed * sigmaz_universe[x][y][z] * time_step) / (permeability_universe[x][y][z] * vacuum_permittivity)) / hf;
				hz_factor4_universe[x][y][z] = ((sigmax_universe[x][y][z] * sigmay_universe[x][y][z] * time_step) / pow(vacuum_permittivity, 2)) / hf;
			}
    	}
	}
}


void handle_perfect_electrical_conductors()
{
	for (int z=0; z < UDEPTH; z++)
	{
		for (int y=0; y < UHEIGHT; y++)
		{
			for (int x=0; x < UWIDTH; x++)
			{
				if (permittivity_universe[x][y][z] == 12345)
				{
					current_ex_universe[x][y][z] = 0.0;
					current_ey_universe[x][y][z] = 0.0;
					current_ez_universe[x][y][z] = 0.0;
				}
			}
		}
	}
}

void handle_perfect_magnetical_conductors()
{
	for (int z=0; z < UDEPTH; z++)
	{
		for (int y=0; y < UHEIGHT; y++)
		{
			for (int x=0; x < UWIDTH; x++)
			{
				if (permeability_universe[x][y][z] == 12345)
				{
					current_hx_universe[x][y][z] = 0.0;
					current_hy_universe[x][y][z] = 0.0;
					current_hz_universe[x][y][z] = 0.0;
				}
			}
		}
	}
}

void update_fields()
{
	// Update the magnetic field
	
	for (int z=1; z < UDEPTH - 1; z++)
	{
		for (int y=1; y < UHEIGHT - 1; y++)
		{
			for (int x=1; x < UWIDTH - 1; x++)
			{
				fp hx_curl_term = ((old_ez_universe[x][y + 1][z] - old_ez_universe[x][y][z]) / grid_width_y) - ((old_ey_universe[x][y][z + 1] - old_ey_universe[x][y][z]) / grid_width_z);
				fp hy_curl_term = ((old_ex_universe[x][y][z + 1] - old_ex_universe[x][y][z]) / grid_width_z) - ((old_ez_universe[x + 1][y][z] - old_ez_universe[x][y][z]) / grid_width_x);
				fp hz_curl_term = ((old_ey_universe[x + 1][y][z] - old_ey_universe[x][y][z]) / grid_width_x) - ((old_ex_universe[x][y + 1][z] - old_ex_universe[x][y][z]) / grid_width_y);
				hx_curl_integral_universe[x][y][z] += hx_curl_term;
				hy_curl_integral_universe[x][y][z] += hy_curl_term;
				hz_curl_integral_universe[x][y][z] += hz_curl_term;
				hx_field_integral_universe[x][y][z] += old_hx_universe[x][y][z];
				hy_field_integral_universe[x][y][z] += old_hy_universe[x][y][z];
				hz_field_integral_universe[x][y][z] += old_hz_universe[x][y][z];
				current_hx_universe[x][y][z] = + hx_factor1_universe[x][y][z] * old_hx_universe[x][y][z] - hx_factor2_universe[x][y][z] * hx_curl_term
											   - hx_factor3_universe[x][y][z] * hx_curl_integral_universe[x][y][z] - hx_factor4_universe[x][y][z] * hx_field_integral_universe[x][y][z];
				current_hy_universe[x][y][z] = + hy_factor1_universe[x][y][z] * old_hy_universe[x][y][z] - hy_factor2_universe[x][y][z] * hy_curl_term
											   - hy_factor3_universe[x][y][z] * hy_curl_integral_universe[x][y][z] - hy_factor4_universe[x][y][z] * hy_field_integral_universe[x][y][z];
				current_hz_universe[x][y][z] = + hz_factor1_universe[x][y][z] * old_hz_universe[x][y][z] - hz_factor2_universe[x][y][z] * hz_curl_term
											   - hz_factor3_universe[x][y][z] * hz_curl_integral_universe[x][y][z] - hz_factor4_universe[x][y][z] * hz_field_integral_universe[x][y][z];
			}
		}
	}
	handle_perfect_magnetical_conductors();


	// Update the electrical field
	for (int z=1; z < UDEPTH - 1; z++)
	{
		for (int y=1; y < UHEIGHT - 1; y++)
		{
			for (int x=1; x < UWIDTH - 1; x++)
			{
				fp ex_curl_term = ((current_hz_universe[x][y][z] - current_hz_universe[x][y - 1][z]) / grid_width_y) - ((current_hy_universe[x][y][z] - current_hy_universe[x][y][z - 1]) / grid_width_z);
				fp ey_curl_term = ((current_hx_universe[x][y][z] - current_hx_universe[x][y][z - 1]) / grid_width_z) - ((current_hz_universe[x][y][z] - current_hz_universe[x - 1][y][z]) / grid_width_x);
				fp ez_curl_term = ((current_hy_universe[x][y][z] - current_hy_universe[x - 1][y][z]) / grid_width_x) - ((current_hx_universe[x][y][z] - current_hx_universe[x][y - 1][z]) / grid_width_y);
				ex_curl_integral_universe[x][y][z] += ex_curl_term;
				ey_curl_integral_universe[x][y][z] += ey_curl_term;
				ez_curl_integral_universe[x][y][z] += ez_curl_term;
				ex_field_integral_universe[x][y][z] += old_ex_universe[x][y][z];
				ey_field_integral_universe[x][y][z] += old_ey_universe[x][y][z];
				ez_field_integral_universe[x][y][z] += old_ez_universe[x][y][z];
				current_ex_universe[x][y][z] = ex_factor1_universe[x][y][z] * old_ex_universe[x][y][z] - ex_factor2_universe[x][y][z] * ex_field_integral_universe[x][y][z]
											   + ex_factor3_universe[x][y][z] * ex_curl_term + ex_factor4_universe[x][y][z] * ex_curl_integral_universe[x][y][z];
				current_ey_universe[x][y][z] = ey_factor1_universe[x][y][z] * old_ey_universe[x][y][z] - ey_factor2_universe[x][y][z] * ey_field_integral_universe[x][y][z]
											   + ey_factor3_universe[x][y][z] * ey_curl_term + ey_factor4_universe[x][y][z] * ey_curl_integral_universe[x][y][z];
				current_ez_universe[x][y][z] = ez_factor1_universe[x][y][z] * old_ez_universe[x][y][z] - ez_factor2_universe[x][y][z] * ez_field_integral_universe[x][y][z]
											   + ez_factor3_universe[x][y][z] * ez_curl_term + ez_factor4_universe[x][y][z] * ez_curl_integral_universe[x][y][z];
			}
		}
	}
	handle_perfect_electrical_conductors();

	int middle_x = (int)UWIDTH / 2;
	int middle_y = (int)UHEIGHT / 2;
	int middle_z = (int)UDEPTH / 2;
	current_ez_universe[middle_x][middle_y][middle_z] = ramped_sinus(current_time, ramped_sin_length * time_step, ramped_sin_amplitude, ramped_sin_frequency);

	memcpy(old_ex_universe, current_ex_universe, sizeof(fp) * UWIDTH * UHEIGHT * UDEPTH);
	memcpy(old_ey_universe, current_ey_universe, sizeof(fp) * UWIDTH * UHEIGHT * UDEPTH);
	memcpy(old_ez_universe, current_ez_universe, sizeof(fp) * UWIDTH * UHEIGHT * UDEPTH);

	memcpy(old_hx_universe, current_hx_universe, sizeof(fp) * UWIDTH * UHEIGHT * UDEPTH);
	memcpy(old_hy_universe, current_hy_universe, sizeof(fp) * UWIDTH * UHEIGHT * UDEPTH);
	memcpy(old_hz_universe, current_hz_universe, sizeof(fp) * UWIDTH * UHEIGHT * UDEPTH);

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

	for (int z = 0; z < UDEPTH; z++)
	{
		for (int y = UHEIGHT / 2; y < UHEIGHT / 2 + 1; y++)
		{
			for (int x = 0; x < UWIDTH; x++)
			{
				// Print as a gnuplot vector x z y dx dy dz
				//printf("%d\t%d\t%d\t%.32f\t%.32f\t%.32f\n", x, y, z, current_ex_universe[x][y][z], current_ey_universe[x][y][z], current_ez_universe[x][y][z]);
				printf("%d\t%d\t%.32f\n", x, z, current_ez_universe[x][y][z]);
			}

		}
	}
	return 0;
}
