#include <opencv4/opencv2/core/hal/interface.h>
#include <random>
#include <stdio.h>

#include "./cilindr.h"

Cilindr::Cilindr(int radius, int number_of_points, int height,
            int center_z, float angle_speed, int z_plate,
            int padding_x, int padding_y, int paint_radius):
            _radius(radius), _height(height), _x(0), _y(0),
            _z(center_z), _z_plate(z_plate), _angle_speed(angle_speed), _angle(0),
            _point_radius(paint_radius) {
    std::random_device devId;
    std::mt19937 randMT; 
    randMT.seed(devId());
    
    std::uniform_int_distribution<int> range_angle(0, 360);
    std::uniform_int_distribution<int> range_height(0, height);

    _points.resize(number_of_points);
    for (point &p : _points) 
        p = {range_angle(randMT), range_height(randMT)};

    _x_left = -float(_x + radius) * _z_plate / (_z - radius) - padding_x;
    int x_right = -float(_x - radius) * _z_plate / (_z - radius) + padding_x;
    _y_down = -float(_y + height / 2) * _z_plate / (_z - radius) - padding_y;
    int y_up = -float(_y - height / 2) * _z_plate / (_z - radius) + padding_y;

    _plate = cv::Mat(y_up - _y_down, x_right - _x_left, CV_8UC1, 255);
}

inline float Deg2Rad(float deg) {
    return deg * std::acos(-1.0) / 180;
}
    
cv::Mat Cilindr::GetNextPhoto() {
    _plate.setTo(255);

    for (point &p : _points) {
        float new_x = _x + _radius * std::cos(Deg2Rad(p.angle + _angle));
        float new_z = _z + _radius * std::sin(Deg2Rad(p.angle + _angle)); 
        float new_y = _y - _height / 2 + p.height;
        int p_x = std::round(-new_x * _z_plate / new_z - _x_left);
        int p_y = std::round(-new_y * _z_plate / new_z - _y_down);
        for (int c_x = p_x - _point_radius; c_x < p_x + _point_radius; ++c_x)
          for (int c_y = p_y - _point_radius; c_y < p_y + _point_radius; ++c_y)
            if ((c_y - p_y) * (c_y - p_y) + (c_x - p_x) * (c_x - p_x) < _point_radius * _point_radius)
              _plate.at<uint8_t>(c_y, c_x) = 0;
    }
    _angle += _angle_speed;
    return _plate;
}