/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	num_particles = 100;
	is_initialized = true;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	double velocity_yaw;
	double yaw_dt;

	bool straight = false;
	if (abs(yaw_rate) < 0.0001)
	{
		straight = true;
	}
	if (!straight)
	{
		velocity_yaw = velocity / yaw_rate;
		yaw_dt = yaw_rate * delta_t;
	}

	default_random_engine gen;
	normal_distribution<double> gaussian_x(0.0, std_pos[0]);
	normal_distribution<double> gaussian_y(0.0, std_pos[1]);
	normal_distribution<double> gaussian_theta(0.0, std_pos[2]);

	for (auto &particle : particles)
	{
		double theta = particle.theta;
		if (!straight)
		{
			particle.x += velocity_yaw * (sin(theta + yaw_dt) - sin(theta));
			particle.y += velocity_yaw * (-cos(theta + yaw_dt) + cos(theta));
			particle.theta += yaw_dt;
		}
		else
		{
			particle.x += velocity * cos(theta) * delta_t;
			particle.y += velocity * sin(theta) * delta_t;
		}

		// add gaussian noise to prediction
		particle.x += gaussian_x(gen);
		particle.y += gaussian_y(gen);
		particle.theta += gaussian_theta(gen);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
																	 std::vector<LandmarkObs> observations, Map map_landmarks)
{
	float max_range = sensor_range * 1.1;

	for (auto &particle : particles)
	{
		std::vector<LandmarkObs> landmarks_in_range;

		for (auto &landmark : map_landmarks.landmark_list)
		{
			std::vector<LandmarkObs> transformed_observations = observations;

			if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= max_range)
			{
				LandmarkObs tmp_landmark;
				tmp_landmark.id = landmark.id_i;
				tmp_landmark.x = landmark.x_f;
				tmp_landmark.y = landmark.y_f;
				landmarks_in_range.push_back(tmp_landmark);
			}

			for (auto &observation : transformed_observations)
			{
				double x_mod = particle.x + cos(particle.theta) * observation.x - sin(particle.theta) * observation.y;
				double y_mod = particle.y + sin(particle.theta) * observation.x + cos(particle.theta) * observation.y;
				observation.x = x_mod;
				observation.y = y_mod;
			}

			dataAssociation(landmarks_in_range, transformed_observations);

			double weight = 1.0;
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];

			double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));

			for (auto &observation : transformed_observations)
			{
				int land_id = observation.id;
				double mu_x = map_landmarks.landmark_list[land_id - 1].x_f;
				double mu_y = map_landmarks.landmark_list[land_id - 1].y_f;
				double dx = observation.x - mu_x;
				double dy = observation.y - mu_y;
				double exponent = (dx * dx) / (2 * sig_x * sig_x) + (dy * dy) / (2 * sig_y * sig_y);
				weight *= gauss_norm * exp(-exponent);
			}
			particle.weight = weight;
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	for (auto &observ : observations)
	{
		double min_dist = 1e15; // initialize with a very big number
		for (unsigned int i = 0; i < predicted.size(); i++)
		{
			LandmarkObs predict = predicted[i];
			if (dist(observ.x, observ.y, predict.x, predict.y) < min_dist)
			{
				observ.id = predict.id;
				min_dist = dist(observ.x, observ.y, predict.x, predict.y);
			}
		}
	}
}

void ParticleFilter::resample()
{
	default_random_engine gen;

	std::vector<double> weights;
	std::vector<Particle> new_particles;

	for (auto &particle : particles)
	{
		weights.push_back(particle.weight);
	}
	std::discrete_distribution<> distribution(weights.begin(), weights.end());

	for (auto &particle : particles)
	{
		new_particles.push_back(particles[distribution(gen)]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
