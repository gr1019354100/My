#ifdef __clang__
#define STBIWDEF static inline
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif // !_USE_MATH_DEFINES

#include <stb/stb_image_write.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

struct Material
{
	Material(
		const cv::Vec3f& _color, float _ks, float _kd, float _a) 
		: diffuse_color(_color), ks(_ks), kd(_kd), alpha(_a) {}

	Material() {}

	cv::Vec3f diffuse_color;
	float ks; // specular reflection constant
	float kd; // diffuse reflection constant
	float ka; // ambient reflection
	float alpha;  // shininess constant
};

struct Light
{
	Light(
		const cv::Vec3f& position, const float& intensive) 
		: m_position(position), is(intensive) {}

	Light() {}

	cv::Vec3f m_position;
	float is; // intensive
};

struct Sphere
{
	cv::Vec3f m_position;
	float m_r;
	Material material;

	Sphere(const cv::Vec3f& position, const float& r, const Material& m) : m_position(position), m_r(r), material(m) {}

	bool ray_intersect(const cv::Vec3f& orig, const cv::Vec3f& dir, float& dis) const
	{
		using namespace cv;
		Vec3f lght_cntr_dist = m_position - orig;
		float len = lght_cntr_dist.dot(dir);
		float dist = lght_cntr_dist.dot(lght_cntr_dist) - len * len;
		if (dist > m_r* m_r) return false;
		float t = sqrtf(m_r * m_r - dist);
		dis = len - t;
		float t1 = len + t;
		if (dis < 0) dist = t1;
		if (dis < 0) return false;
		return true;
	}
};

bool scene_intersect(const cv::Vec3f& orig, const cv::Vec3f& dir, const std::vector<Sphere>& spheres,
	cv::Vec3f& hit, cv::Vec3f& N, Material& material)
{
	float sphere_dist = std::numeric_limits<float>::max();
	for (int i = 0; i < spheres.size(); i++)
	{
		float dist_i;
		if (spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < sphere_dist)
		{
			sphere_dist = dist_i;
			hit = orig + dir * dist_i;
			N = normalize((hit - spheres[i].m_position));
			material = spheres[i].material;
		}
	}
	return sphere_dist < 1000;
}

cv::Vec3f reflect(const cv::Vec3f& I, const cv::Vec3f& N) {
	return cv::normalize(I - N * 2.f * (I * N));
}

cv::Vec3f cast_ray(const cv::Vec3f& orig,
	const cv::Vec3f& dir, const std::vector<Sphere>& spheres, const std::vector<Light>& lights)
{
	cv::Vec3f point, N;
	Material material;
	if (!scene_intersect(orig, dir, spheres, point, N, material)) {
		return cv::Vec3f(0.2, 0.7, 0.8);
	}
	float diffuse_light_intensity = 0;
	float specular_light_intensity = 0;
	for (int i = 0; i < lights.size(); i++)
	{
		// diffuse reflection
		cv::Vec3f light_dir = normalize((lights[i].m_position - point));

		float light_distance = cv::norm((lights[i].m_position - point));

		cv::Vec3f shadow_orig = light_dir.dot(N) < 0 ? point - N * 1e-3 : point + N * 1e-3; // checking if the point lies in the shadow of the lights[i]
		cv::Vec3f shadow_pt, shadow_N;
		Material tmpmaterial;
		if (scene_intersect(shadow_orig, light_dir, spheres, shadow_pt, shadow_N, tmpmaterial) && cv::norm(shadow_pt - shadow_orig) < light_distance)
			continue;

		diffuse_light_intensity += lights[i].is * std::max(0.f, light_dir.dot(N));

		// phong reflection on sphere

		specular_light_intensity += powf(std::max(0.f, -reflect(-light_dir, N).dot(dir)), material.alpha) * lights[i].is;
	}
	return material.diffuse_color * diffuse_light_intensity * material.kd
		+ cv::Vec3f(1.f, 1.f, 1.f) * specular_light_intensity * material.ks;
}

void render(const std::vector<Sphere>& spheres, const std::vector<Light>& lights)
{
	unsigned char* tframebuffer;
	const int width = 1024;
	const int height = 768;
	const float fov = M_PI_4;
	std::vector<cv::Vec3f> framebuffer(width * height);
	tframebuffer = new unsigned char[3 * width * height];
#pragma omp parallel for
	for (unsigned int j = 0; j < height; j++)
		for (unsigned int i = 0; i < width; i++)
		{
			float x = (2 * (i + 0.5) / (float)width - 1) * tan(fov / 2.) * width / (float)height;
			float y = -(2 * (j + 0.5) / (float)height - 1) * tan(fov / 2.);
			cv::Vec3f dir = cv::normalize(cv::Vec3f(x, y, -1));
			framebuffer[i + j * width] = cast_ray(cv::Vec3f(0, 0, 0), dir, spheres, lights);
		}
	for (int i = 0, j = -1; i < framebuffer.size(); i++)
	{
		cv::Vec3f& c = framebuffer[i];
		float max = std::max(c[0], std::max(c[1], c[2]));
		if (max > 1.) c = c * (1. / max);
		tframebuffer[++j] = (unsigned char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][0])));
		tframebuffer[++j] = (unsigned char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][1])));
		tframebuffer[++j] = (unsigned char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][2])));
	}
	stbi_write_jpg("C:\\Users\\gr199\\Desktop\\test_shadow.jpg", width, height, 3, tframebuffer, 100);
	delete[] tframebuffer;
}

int main()
{
	Material ivory_arylic_glass(cv::Vec3f(0.4, 0.4, 0.3), 0.3, 0.6, 50.);
	Material red_rubber(cv::Vec3f(0.3, 0.1, 0.1), 0.1, 0.9, 10.);

	std::vector<Sphere> spheres;
	spheres.push_back(Sphere(cv::Vec3f(-3, 0, -16), 2, ivory_arylic_glass));
	spheres.push_back(Sphere(cv::Vec3f(-1.0, -1.5, -12), 2, red_rubber));
	spheres.push_back(Sphere(cv::Vec3f(1.5, -0.5, -18), 3, ivory_arylic_glass));
	spheres.push_back(Sphere(cv::Vec3f(7, 5, -18), 4, red_rubber));

	std::vector<Light>  lights;
	lights.push_back(Light(cv::Vec3f(-20, 20, 20), 1.5));
	lights.push_back(Light(cv::Vec3f(30, 50, -25), 1.8));
	lights.push_back(Light(cv::Vec3f(30, 20, 30), 1.7));

	render(spheres, lights);
	return 0;
}