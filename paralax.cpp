#include <cilindr.h>
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
  Solver(): 
  _solved(false) {};

  void Reset() {
    _solved = false;  
    _imgs.clear();
    _H.clear();
    _Z.clear();
    _angles.clear();
    _angle_speed.clear();
    _z_plate.clear();
  }

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
      UpdateAll(_imgs[2]);
      

      _imgs[2].copyTo(tmp_img);
      Solve().copyTo(tmp_img);
      
      if (_solved) {
        putText(tmp_img, std::to_string(mean(_angle_speed)), cv::Point(30,30), 
            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0), 1, 0);
        DrawALL(tmp_img);
      }


      return tmp_img;
    }
    return img;
  }


private:
  void DrawALL(cv::Mat &img) {
    for (size_t i = 0; i < _angle_speed.size(); ++i)
      EmphasizePoint(img, GetPoint(_H[i], _Z[i], _angles[i], _z_plate[i]), 0).copyTo(img);
  }

  void UpdateAll(const cv::Mat &img) {
    for (size_t i = 0; i < _angles.size(); ++i)
      _angles[i] += _angle_speed[i] * std::acos(-1.f) / 180;
    
    int r = 4;
    std::vector<int> trash;
    
    for (size_t i = 0; i < _angle_speed.size(); ++i) {
      pt p = GetPoint(_H[i], _Z[i], _angles[i], _z_plate[i]);
      bool f = true;
      for (size_t x = std::max(p.x - r, 0); x < std::min(p.x + r, img.cols) && f; ++x)
        for (size_t y = std::max(p.y - r, 0); y < std::min(p.y + r, img.rows) && f; ++y)
          f &= img.at<uint8_t>(y, x) > 0;
      if (f)
        trash.push_back(i);
    }

    for (int stride=0, i = 0, j = 0; i < _angles.size() - stride; ++i) {
      if (j < trash.size() && i + stride == trash[j]) {
        j += 1;
        stride += 1;
        i -= 1;
      } else {
        _z_plate[i] = _z_plate[i + stride];
        _H[i] = _H[i + stride];
        _Z[i] = _Z[i + stride];
        _angles[i] = _angles[i + stride];
        _angle_speed[i] = _angle_speed[i + stride];
      }
    }

    _z_plate.resize(_z_plate.size() - trash.size());
    _H.resize(_H.size() - trash.size());
    _Z.resize(_Z.size() - trash.size());
    _angles.resize(_angles.size() - trash.size());
    _angle_speed.resize(_angle_speed.size() - trash.size());

    _solved = _z_plate.size();
  }

  pt GetPoint(float H, float Z, float alpha, float z_plate) {
    float new_x = std::cos(alpha);
    float new_z = Z - std::sin(alpha); 
    float new_y = H;
    int p_x = std::round(new_x * z_plate / new_z)  + _imgs[2].cols / 2;
    int p_y = std::round(new_y * z_plate / new_z)  + _imgs[2].rows / 2;
    return {p_x, p_y};
  }

  float mean(std::vector<float> ans) {
    std::sort(ans.begin(), ans.end());
    return ans[ans.size() / 2];
    float v = 0;
    for (const float &a : ans)
      v += a;
    return v / ans.size();
  }

  void CleanPoint(cv::Mat &cur, pt point) {
    int dirx[] = {0, 1, 0, -1};
    int diry[] = {1, 0, -1, 0};
    std::vector<pt> p;
    p.push_back(point);
    size_t c = 0;
    while (c != p.size()) {
      pt &p_cur = p[c++];
      if (cur.at<uint8_t>(p_cur.y, p_cur.x) != 0) 
        continue;
      cur.at<uint8_t>(p_cur.y, p_cur.x) = 255;
      for (size_t i = 0; i < 4; ++i) {
        if (p_cur.x + dirx[i] < 0 || p_cur.x + dirx[i] > cur.cols)
          continue;
        if (p_cur.y + diry[i] < 0 || p_cur.y + diry[i] > cur.rows)
          continue;
        if (cur.at<uint8_t>(p_cur.y + diry[i], p_cur.x + dirx[i]) == 0)
          p.push_back({p_cur.x + dirx[i], p_cur.y + diry[i]});
      }
    } 
  }

  std::vector<pt> FindTop(const cv::Mat &img, size_t n) {
    cv::Mat cur;
    img.copyTo(cur);
    std::vector<pt> up;

    for (int y = 0; up.size() < n && y < img.rows; ++y)
        for (int x = 0; x < img.cols && up.size() < n; ++x) 
          if (cur.at<uint8_t>(y, x) == 0) {
            up.push_back(pt({x, y}));
            CleanPoint(cur, pt({x, y}));
          }
    
    return up;
  }

  float GetAngleSpeed(float &speed, std::vector<pt> p, int w, int h) {
    for (pt &pp : p) {
      pp.x -= w / 2;
      pp.y -= h / 2;
    }

    float y0 = p[0].y, y1 = p[1].y, y2 = p[2].y;
    float x0 = p[0].x, x1 = p[1].x, x2 = p[2].x;
    // printf("%f %f %f %f %f %f\n", x0, y0, x1, y1, x2, y2);

    if (!(x0 > x1 && x1 > x2) && !(x0 < x1 && x1 < x2))
      return false;

    if (fabs(x1 * y2 * y0 * x2) < 1e-5)
      return false;

    float v = 0.5 * ((y1 * x0) / (x1 * y0) + (x2 * y1) / (x1 * y2));
    
    if (abs(v) >= 1)
      return false;

    float rad_speed = std::acos(v);
    if (x1 < x2)
      rad_speed = -rad_speed;

    speed = rad_speed * 180 / std::acos(-1.0);
    if (abs(speed) > 90)
      return false;

    float alpha = 0.0;
    bool is_ok=false;
    float z_plate;

    for (size_t i = 1; i < 3; ++i) {
      if (p[i].x * p[0].y != 0) {
        float e = (p[0].x * p[i].y) / (p[i].x * p[0].y);
        if (e * std::sin(rad_speed * i) > 1e-5) {
          alpha = std::atan((e * std::cos(rad_speed * i) - 1) /
          (e * std::sin(rad_speed * i)));
          z_plate = (std::sin(alpha + rad_speed * i) - std::sin(alpha)) / (std::cos(alpha) / p[i].x - std::cos(alpha) / p[0].x);
          is_ok = true;
        }
      }
    }

    if (!is_ok)
      return false;

    if (std::sin(alpha + rad_speed * 2) == 0)
      return false;

    float H = y2 * std::cos(alpha + rad_speed * 2) / x2;
    float Z = std::cos(alpha + rad_speed * 2) * z_plate / x2 + std::sin(alpha + rad_speed * 2);

    if (Z <= 1) {
      return false; 
    }

    _angles.push_back(alpha + rad_speed * 2);
    _H.push_back(H);
    _angle_speed.push_back(speed);
    _Z.push_back(Z);
    _z_plate.push_back(z_plate);
    printf("%f %f %f %f\n", mean(_angles), H, Z, z_plate);

    return true;
  }

  cv::Mat EmphasizePoint(const cv::Mat image, pt p, uint8_t color) {
    cv::Mat im_to_draw;
    image.copyTo(im_to_draw);
    for (size_t x = p.x - 10; x < p.x + 10; ++x)
      for (size_t y = p.y - 10; y < p.y - 8; ++y)
        im_to_draw.at<uint8_t>(y, x) = color;

    for (size_t x = p.x - 10; x < p.x + 10; ++x)
      for (size_t y = p.y + 8; y < p.y + 10; ++y)
        im_to_draw.at<uint8_t>(y, x) = color;

    for (size_t y = p.y - 10; y < p.y + 10; ++y)
      for (size_t x = p.x + 8; x < p.x + 10; ++x)
        im_to_draw.at<uint8_t>(y, x) = color;

    for (size_t y = p.y - 10; y < p.y + 10; ++y)
      for (size_t x = p.x - 10; x < p.x - 8; ++x)
        im_to_draw.at<uint8_t>(y, x) = color;

    return im_to_draw;
  }

  cv::Mat Solve() {
    std::vector<pt> up[3];
    int pt_fr = 1;
    
    cv::Mat draw_img;
    _imgs[2].copyTo(draw_img);   

    for (size_t i = 0; i < 3; ++i) {
      up[i] = FindTop(_imgs[i], pt_fr);    
    }
    EmphasizePoint(draw_img, up[2][0], 120).copyTo(draw_img);


    for (size_t i = 0, j = 0, k = 0; i < pt_fr; k += 1, j += k / pt_fr, i += j / pt_fr, k %=pt_fr, j %= pt_fr) {
      float speed;
      if (GetAngleSpeed(speed, {up[0][i], up[1][j], up[2][k]}, _imgs[2].cols, _imgs[2].rows)) {
        EmphasizePoint(draw_img, up[2][k], 0).copyTo(draw_img);
        break;
      }
    }

    _solved |= _angle_speed.size();
    return draw_img;
  }

public:
  bool _solved;  
  std::vector<cv::Mat> _imgs;
  std::vector<float> _H;
  std::vector<float> _Z;
  std::vector<float> _angles;
  std::vector<int> _iteration;
  std::vector<float> _angle_speed;
  std::vector<float> _z_plate;
};

int main() {
    Cilindr a(50, 50, 100, 100, 10, 90, 40, 40, 1);
    Solver s;
    for (size_t i = 0; i < 450; ++i) {
      cv::Mat im = a.GetNextPhoto();
      std::string name = "/mnt/c/Users/1/Desktop/1/test" + std::to_string(i) + ".png";
      s.AddImage(im).copyTo(im);
      cv::imwrite(name.c_str(), im);
      // if (s._solved)
      //   return 0;
    }
}