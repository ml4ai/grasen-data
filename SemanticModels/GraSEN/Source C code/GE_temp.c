#include <stdio.h>
#include <stdbool.h>

#define DEADZONE_UPPER 100
#define DEADZONE_LOWER -100

double integrator_state = 0.0;
double upper_threshold = 1100.0;
double lower_threshold = 0.0;

/*
 * Simplified version of the T-MATS Simple PI controller library block
 */
double PI_calc(double Input_dmd, double Input_sensed, double Kp_M, double Ki_M, double timestep)
{
    double error = Input_dmd - Input_sensed;
    bool update_integrator;
	double u;
	update_integrator = true; 
	u = error*Kp_M + (integrator_state + timestep*error)*Ki_M;
	if (u > upper_threshold) // positive saturation
	{ // IF_0
		u = upper_threshold; // clamp the output
		if (error > 0) // if error is the same sign then inhibit integration
		{ // IF_0.IF_1
			update_integrator = false;
		};
	}
	else if (u < lower_threshold) // repeat for negative sign (also else if)
	{ // IF_0.IF_2
		u = lower_threshold;
		if (error < 0) // IF_0.IF_2.IF_3
		{
			update_integrator = false;
		};
	};
	if (update_integrator) // IF 4 
	{
		integrator_state = integrator_state + timestep*error;
	}
    return u;
}

/*
 * Simple deadzone function
 */
 // AM: the following runs fine but if I uncomment the 2 lines (53-54) then I get GrFN error
/*double deadzone_function(double num1)
{
	double x;
    if (num1 > DEADZONE_UPPER) {
        x = num1 - DEADZONE_UPPER;
    } 
    if (num1 <= DEADZONE_UPPER && num1 >= DEADZONE_LOWER) {
        x = 0.0;
    } 
    if (num1 < DEADZONE_LOWER){
        x = num1 - DEADZONE_LOWER;
	}
	return x;
}*/

/*
 * Proportional plant!
 */
double plant_model(double input, double gain)
{
    return input*gain; //10 --> since gain 0.01, input must be 1000
}

int main(int argc, char **argv)
{

    double t_final = 100.5;
    double time_step = 0.015;
    
    double Ki_M = 20.0;
    double Kp_M = 75.0;
	
	int num_steps = t_final / time_step;
    	
	double desired_output = 10.0;
	
	double plant_command;
	double sensed_output;

    double plant_gain = 0.01;

	sensed_output = 0.0;

    //FILE *fptr;
    //fptr = fopen("C:\\Users\\200015853\\Downloads\\output_data.txt","w");

    for (int i = 0; i < num_steps; i++)
    {
        plant_command = PI_calc(desired_output, sensed_output, Kp_M, Ki_M, time_step);
		
		sensed_output = plant_model(plant_command, plant_gain);

        //fprintf(fptr,"%f, %f, %f\n", (double)i*time_step, plant_command, sensed_output);
        printf("%f, %f, %f\n", (double)i*time_step, plant_command, sensed_output);
    }

    //fclose(fptr);

    return 0;
}
