#include <stdio.h>

struct _pid
{
	float SetSpeed;
	float ActualSpeed;
	float err;
	float err_last;
	float derv;
	float Kp, Ki, Kd;
	float voltage;
	float integral;
} pid;

void PID_init()
{
	printf("PID_init begin \n");
	pid.SetSpeed = 0.0;
	pid.ActualSpeed = 0.0;
	pid.err = 0.0;
	pid.err_last = 0.0;
	pid.voltage = 0.0;
	pid.integral = 0.0;
	pid.derv = 0.0;
	pid.Kp = 0.2;
	pid.Ki = 0.015;
	pid.Kd = 0.2;
	printf("PID_init end \n");
}

float PID_realize(float speed)
{
	pid.SetSpeed = speed;
	pid.err = pid.SetSpeed - pid.ActualSpeed;
	pid.integral += pid.err;
	pid.derv = pid.err - pid.err_last;
	pid.voltage = pid.Kp * pid.err + pid.Ki * pid.integral + pid.Kd * pid.derv;
	pid.err_last = pid.err;
	pid.ActualSpeed = pid.voltage * 1.0;
	return pid.ActualSpeed;
}

int main()
{
	printf("System begin \n");
	PID_init();
	int count = 0;
	while (count < 100)
	{
		float speed = PID_realize(20.0);
		printf("%f\n", speed);
		count++;
	}
	return 0;
}
