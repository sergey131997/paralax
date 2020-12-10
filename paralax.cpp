#include "cilindr.h"
#include <algorithm>
#include <opencv4/opencv2/core/hal/interface.h>
#include <string>

// x = (R * cos(a + d) ) / (Z + R * sin(a + d) ) 
// y = height / (Z + R * sin(a + d) )
// R, a, d, height, Z

struct pt {
  int x;
  int y;
};

class Solver {
public:
  Solver(int z_plate = 100): 
  _solved(false), _z_plate(z_plate) {};

  cv::Mat AddImage(const cv::Mat &img) {
    if (_imgs.size() < 3) {
      _imgs.push_back(cv::Mat());
      img.copyTo(_imgs[_imgs.size() - 1]);
    }
    else {
      for (size_t i = 0; i < 2; ++i)
        std::swap(_imgs[i], _imgs[i + 1]);
      img.copyTo(_imgs[2]);
      
      cv::Mat tmp_img;
      Solve().copyTo(tmp_img);
      if (_solved) {
        putText(tmp_img, std::to_string(mean(_angle_speed)), cv::Point(30,30), 
            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0), 1, 0);
      }


      return tmp_img;
    }
    return img;
  }
private:
  float mean(std::vector<float> ans) {
    std::sort(ans.begin(), ans.end());
    return ans[ans.size() / 2];
    float v = 0;
    for (const float &a : ans)
      v += a;
    return v / ans.size();
  }
   
  std::pair<pt, pt> FindTop(const cv::Mat &img) {
    pt up({-1, -1});
    pt down({-1, -1});
    for (int y = 0; up.x == -1; ++y)
        for (int x = 0; x < img.cols && up.x == -1; ++x) 
          if (img.at<uint8_t>(y, x) == 0) 
            up = pt({x, y});
    
    for (int y = img.rows - 1; down.x == -1; --y)
        for (int x = 0; x < img.cols && down.x == -1; ++x) 
          if (img.at<uint8_t>(y, x) == 0) 
            down = pt({x, y});
    return {down, up};          
  }

  bool GetAngleSpeed(float &speed, pt p0, pt p1, pt p2, int w, int h) {
    float y0 = p0.y - h / 2, y1 = p1.y - h / 2, y2 = p2.y - h / 2;
    float x0 = p0.x - w / 2, x1 = p1.x - w / 2, x2 = p2.x - w / 2;

    if (!(x0 > x1 && x1 > x2) && !(x0 < x1 && x1 < x2))
      return false;

    if (fabs(x1 * y2 * y0) < 1e-5)
      return false;

    float v = 0.5 * ((y1 * x0) / (x1 * y0) + (x2 * y1) / (x1 * y2));
    
    if (v >= 1)
      return false;

    speed = std::acos(v) * 180 / std::acos(-1.0);

    return (abs(speed) < 90);
  }

  cv::Mat EmphasizePoint(const cv::Mat image, pt p) {
    cv::Mat im_to_draw;
    image.copyTo(im_to_draw);
    for (size_t x = p.x - 10; x < p.x + 10; ++x)
      for (size_t y = p.y - 10; y < p.y - 8; ++y)
        im_to_draw.at<uint8_t>(y, x) = 0;

    for (size_t x = p.x - 10; x < p.x + 10; ++x)
      for (size_t y = p.y + 8; y < p.y + 10; ++y)
        im_to_draw.at<uint8_t>(y, x) = 0;

    for (size_t y = p.y - 10; y < p.y + 10; ++y)
      for (size_t x = p.x + 8; x < p.x + 10; ++x)
        im_to_draw.at<uint8_t>(y, x) = 0;

    for (size_t y = p.y - 10; y < p.y + 10; ++y)
      for (size_t x = p.x - 10; x < p.x - 8; ++x)
        im_to_draw.at<uint8_t>(y, x) = 0;

    return im_to_draw;
  }

  cv::Mat Solve() {
    pt up[3];
    pt down[3];
    for (size_t i = 0; i < 3; ++i) {
      std::pair<pt, pt> ans = FindTop(_imgs[i]);
      down[i] = ans.first;
      up[i] = ans.second;
    }

    cv::Mat draw_img;
    _imgs[2].copyTo(draw_img);   

    float speed0, speed1;
    if (GetAngleSpeed(speed0, up[0], up[1], up[2],  _imgs[0].rows, _imgs[0].cols) && 
       GetAngleSpeed(speed1, down[0], down[1], down[2],  _imgs[0].rows, _imgs[0].cols)) 
          if (std::abs(speed0 - speed1) < 5) {
            _angle_speed.push_back((speed0 + speed1) / 2);
            EmphasizePoint(draw_img, up[2]).copyTo(draw_img);
            EmphasizePoint(draw_img, down[2]).copyTo(draw_img);
          } else {
            printf("%f %f\n", speed0, speed1);
          }

    _solved |= _angle_speed.size();
    return draw_img;
  }


  bool _solved;  
  std::vector<cv::Mat> _imgs;
  float _R;
  float _H;
  std::vector<float> _angles;
  std::vector<float> _heights;
  std::vector<float> _angle_speed;
  float _z_plate;
};

int main() {
    Cilindr a(50, 100, 100, 0, 0, 80, 25, 180, 40, 30, 2);
    Solver s;
    for (size_t i = 0; i < 450; ++i) {
      cv::Mat im = a.GetNextPhoto();
      std::string name = "/mnt/c/Users/1/Desktop/1/test" + std::to_string(i) + ".png";
      s.AddImage(im).copyTo(im);
      cv::imwrite(name.c_str(), im);
    }
}