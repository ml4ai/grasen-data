#include <stdio.h>

double integrator_state = 0.0;
double derv = 0.0;
double error_last1 = 0.0;
double error_last2 = 0.0;
double error_last3 = 0.0;

double Comp1(double Input_dmd, double Input_sensed, double Kp_M, double Ki_M, double timestep)
{
    error_last1 = Input_dmd - Input_sensed; // negative example of PrevVal
	integrator_state = integrator_state + timestep*error_last1; // negative example of PrevVal
    return error_last1*Kp_M + integrator_state*Ki_M;
}

double Comp2(double Input_dmd, double Input_sensed, double Kp_M, double Ki_M, double Kd_M, double timestep)
{
  double error = Input_dmd - Input_sensed; // negative example of PrevVal
  integrator_state = integrator_state + timestep*error; // negative example of PrevVal
  derv = error - error_last2; // negative example of PrevVal
  error_last2 = error;  //positive example of PrevVal
  return error_last2*Kp_M + integrator_state*Ki_M + derv*Kd_M;
}

double Comp3(double Input_dmd, double Input_sensed, double Kp_M, double Ki_M, double Kd_M, double timestep)
{
  double error = Input_dmd + Input_sensed;  // negative example of PrevVal
  error_last3 = error;  //positive example of PrevVal
  return error_last3*Kp_M + integrator_state*Ki_M + derv*Kd_M;
}

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
  double Kd_M = 10.0;	
  int num_steps = t_final / time_step;    	
  double desired_output = 10.0;	
  double plant_command;
  double sensed_output = 5.0;	// temp
  double plant_gain = 0.01;

  sensed_output = 0.0;

  for (int i = 0; i < num_steps; i++)
    {
      plant_command = Comp1(desired_output, sensed_output, Kp_M, Ki_M, time_step);
	  plant_command = Comp2(desired_output, sensed_output, Kp_M, Ki_M, Kd_M, time_step);
//      plant_command = Comp3(desired_output, sensed_output, Kp_M, Ki_M, Kd_M, time_step);	  
		
//      sensed_output = plant_model(plant_command, plant_gain);

//      printf("%f, %f, %f\n", (double)i*time_step, plant_command, sensed_output);
    }
  return 0;
}
