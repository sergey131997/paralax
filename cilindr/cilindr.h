#ifndef CILINDR_H
#define CILINDR_H

#include <opencv4/opencv2/opencv.hpp>
#include <vector>


struct point {
  int angle;
  int height;
  point(int angle, int h): angle(angle), height(h) {};
  point(): angle(0), height(0) {};
};
  

class Cilindr {
public:
    Cilindr(int radius, int number_of_points, int height,
            int center_z, float angle_speed, int z_plate,
            int padding_x = 10, int padding_y = 10, int point_radius = 2);
    
    cv::Mat GetNextPhoto();
private:
  int _radius;
  int _angle;
  int _height;
  int _x;
  int _y;
  int _z;
  int _z_plate;
  int _x_left;
  int _y_down;
  float _angle_speed;
  int _point_radius;
  std::vector<point> _points;
  cv::Mat _plate;
};

#endif // CILINDR_H