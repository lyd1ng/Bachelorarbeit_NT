#ifndef fp
#define fp float
#endif

#define UWIDTH 130
#define UHEIGHT 130


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fdtd_helper.h"

fp old_ez_universe[UWIDTH][UHEIGHT] = {0};
fp old_hx_universe[UWIDTH][UHEIGHT] = {0};
fp old_hy_universe[UWIDTH][UHEIGHT] = {0};
fp current_ez_universe[UWIDTH][UHEIGHT] = {0};
fp current_hx_universe[UWIDTH][UHEIGHT] = {0};
fp current_hy_universe[UWIDTH][UHEIGHT] = {0};
fp hx_integral_universe[UWIDTH][UHEIGHT] = {0}; 
fp ez_integral_universe[UWIDTH][UHEIGHT] = {0};
fp hy_integral_universe[UWIDTH][UHEIGHT] = {0};
fp sigmax_universe[UWIDTH][UHEIGHT] = {0};
fp sigmay_universe[UWIDTH][UHEIGHT] = {0};
fp permittivity_universe[UWIDTH][UHEIGHT] = {0};
fp permeability_universe[UWIDTH][UHEIGHT] = {0};

fp e_factor1_universe[UWIDTH][UHEIGHT] = {0};             
fp e_factor2_universe[UWIDTH][UHEIGHT] = {0};             
fp e_factor3_universe[UWIDTH][UHEIGHT] = {0};             
fp hx_factor1_universe[UWIDTH][UHEIGHT] = {0}; 
fp hx_factor2_universe[UWIDTH][UHEIGHT] = {0};
fp hx_factor3_universe[UWIDTH][UHEIGHT] = {0};
fp hy_factor1_universe[UWIDTH][UHEIGHT] = {0};
fp hy_factor2_universe[UWIDTH][UHEIGHT] = {0};
fp hy_factor3_universe[UWIDTH][UHEIGHT] = {0};

fp fraction;
int samplerate_space;
int samplerate_time;
fp grid_width_x;
fp grid_width_y;
fp time_step;
fp current_time;
fp max_time;

fp ramped_sin_frequency;
fp ramped_sin_amplitude;
int ramped_sin_length;



void print_simulation_parameter()
{
	printf("# dimensions: %dx%d\n", UWIDTH, UHEIGHT);
	printf("# space_sr: %f\n", samplerate_time);
	printf("# time_sr: %f\n", samplerate_space);
	printf("# floating point: %d\n", 8 * sizeof(fp));
	printf("# ramped_sin_frequency: %f\n", ramped_sin_frequency);
	printf("# ramped_sin_amplitude: %f\n", ramped_sin_amplitude);
	printf("# ramped_sin_length: %d\n", ramped_sin_length);
	printf("# ramped_sin_length: %.32f\n", max_time);
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
	time_step = get_time_resolution_2d(fraction, grid_width_x, grid_width_y) / (fp)samplerate_time;
	current_time = 0;
	max_time = 10000 * time_step;

	for (int y=0; y < UHEIGHT; y++)
	{
		for (int x=0; x < UWIDTH; x++)
		{
			permittivity_universe[x][y] = 1;
			permeability_universe[x][y] = 1;
		}
	}
}


void set_pml_geometry()
{
    int pml_width = 40;
    int pml_height = 40;
    fp f = vacuum_permittivity / (2.0 * time_step);
    for (int y=1; y < UHEIGHT - 1; y++)
    {
        for (int x=1; x < pml_width; x++)
	{
            sigmax_universe[pml_width - x - 1][y] = f * pow((fp)x / (fp)pml_width, 3);
            sigmax_universe[UWIDTH - pml_width + x][y] = f * pow((fp)x / (fp)pml_width, 3);
	}
    }
    for (int x=1; x < UWIDTH - 1; x++)
    {
        for (int y=1; y < pml_height; y++)
	{
            sigmay_universe[x][pml_height - y - 1] = f * pow((fp)y / pml_height, 3);
            sigmay_universe[x][UHEIGHT - pml_height + y] = f * pow((fp)y / pml_height, 3);
	}
    }
}

void calculate_factors()
{
    for (int x=0; x < UWIDTH; x++)
    {
        for (int y=0; y < UHEIGHT; y++)
	{
		fp nominator = 0;
		fp denominator = 0;
            	e_factor1_universe[x][y] = 1 - ((time_step * (sigmax_universe[x][y] + sigmay_universe[x][y])) / vacuum_permittivity);
            	e_factor2_universe[x][y] = (sigmax_universe[x][y] * sigmay_universe[x][y] * pow(time_step, 2)) / pow(vacuum_permittivity, 2);
            	e_factor3_universe[x][y] = (light_speed * time_step) / permittivity_universe[x][y];

            	hx_factor1_universe[x][y] = (1 / time_step - (sigmay_universe[x][y] / (2 * vacuum_permittivity))) / (1 / time_step + (sigmay_universe[x][y] / (2 * vacuum_permittivity)));
            	hx_factor2_universe[x][y] = (light_speed / permittivity_universe[x][y]) / ((1 / time_step) + (sigmay_universe[x][y] / (2 * vacuum_permittivity)));
            	nominator = (light_speed * sigmax_universe[x][y] * time_step) / (permeability_universe[x][y] * vacuum_permittivity);
            	denominator = 1 / time_step + (sigmay_universe[x][y] / (2 * vacuum_permittivity));
            	hx_factor3_universe[x][y] = nominator / denominator;

            	hy_factor1_universe[x][y] = (1 / time_step - (sigmax_universe[x][y] / (2 * vacuum_permittivity))) / (1 / time_step + (sigmax_universe[x][y] / (2 * vacuum_permittivity)));
            	hy_factor2_universe[x][y] = (light_speed / permittivity_universe[x][y]) / ((1 / time_step) + (sigmax_universe[x][y] / (2 * vacuum_permittivity)));
            	nominator = (light_speed * sigmay_universe[x][y] * time_step) / (permeability_universe[x][y] * vacuum_permittivity);
            	denominator = 1 / time_step + (sigmax_universe[x][y] / (2 * vacuum_permittivity));
            	hy_factor3_universe[x][y] = nominator / denominator;
	}
    }
}


void handle_perfect_electrical_conductors()
{
	for (int x=0; x < UWIDTH; x++)
	{
		for (int y=0; y < UHEIGHT; y++)
		{
			if (permittivity_universe[x][y] == 12345)
			{
				current_ez_universe[x][y] = 0.0;
			}
		}
	}
}

void handle_perfect_magnetical_conductors()
{
	for (int x=0; x < UWIDTH; x++)
	{
		for (int y=0; y < UHEIGHT; y++)
		{
			if (permeability_universe[x][y] == 12345)
			{
				current_hx_universe[x][y] = 0.0;
				current_hy_universe[x][y] = 0.0;
			}
		}
	}
}

void update_fields()
{
	// Update the magnetic field
	for (int y=1; y < UHEIGHT - 1; y++)
	{
		for (int x=1; x < UWIDTH - 1; x++)
		{
			fp hx_curl_term = ((old_ez_universe[x][y + 1] - old_ez_universe[x][y]) / grid_width_y);
			fp hy_curl_term = ((old_ez_universe[x + 1][y] - old_ez_universe[x][y]) / grid_width_x);
			hx_integral_universe[x][y] += hx_curl_term;
			hy_integral_universe[x][y] += hy_curl_term;
			current_hx_universe[x][y] = hx_factor1_universe[x][y] * old_hx_universe[x][y] - hx_factor2_universe[x][y] * hx_curl_term - hx_factor3_universe[x][y] * hx_integral_universe[x][y];
			current_hy_universe[x][y] = hy_factor1_universe[x][y] * old_hy_universe[x][y] + hy_factor2_universe[x][y] * hy_curl_term + hy_factor3_universe[x][y] * hy_integral_universe[x][y];
		}
	}
	handle_perfect_magnetical_conductors();


	// Update the electrical field
	for (int y=1; y < UHEIGHT - 1; y++)
	{
		for (int x=1; x < UWIDTH - 1; x++)
		{
			ez_integral_universe[x][y] += e_factor2_universe[x][y] * old_ez_universe[x][y];
			fp curl_term = ((current_hy_universe[x][y] - current_hy_universe[x - 1][y]) / grid_width_x) - ((current_hx_universe[x][y] - current_hx_universe[x][y - 1]) / grid_width_y);
            		current_ez_universe[x][y] = e_factor1_universe[x][y] * old_ez_universe[x][y] - ez_integral_universe[x][y] + e_factor3_universe[x][y] * curl_term;
		}
	}
	handle_perfect_electrical_conductors();

	int middle_x = (int)UWIDTH / 2;
	int middle_y = (int)UHEIGHT / 2;
	current_ez_universe[middle_x][middle_y] = ramped_sinus(current_time, ramped_sin_length, ramped_sin_amplitude, ramped_sin_frequency);

	memcpy(old_ez_universe, current_ez_universe, sizeof(fp) * UWIDTH * UHEIGHT);
	memcpy(old_hx_universe, current_hx_universe, sizeof(fp) * UWIDTH * UHEIGHT);
	memcpy(old_hy_universe, current_hy_universe, sizeof(fp) * UWIDTH * UHEIGHT);
}

int main()
{
	init_params();
	set_pml_geometry();
	calculate_factors();
	while (current_time < max_time)
	{
		update_fields();
		current_time += time_step;
		
	}

	print_simulation_parameter();
	for (int y = 0; y < UHEIGHT; y++)
	{
		for (int x = 0; x < UWIDTH; x++)
		{
			printf("%d\t%d\t%.32f\n", x, y, current_ez_universe[x][y]);
		}

	}
	return 0;
}
