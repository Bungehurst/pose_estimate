#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // use this if you are in OpenCV2
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> match;
  //BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

Mat vec2rot(const Mat &v){
    Mat rot;
    rot =  (Mat_<double>(3,3) << 
            0, -v.at<double>(2,0), v.at<double>(1,0),
            v.at<double>(2,0), 0, -v.at<double>(0,0),
           -v.at<double>(1,0),v.at<double>(0,0),0);
    return rot;
}

Point2d pix2cam(const Point2d &p, const Mat &k){
    return Point2d(
        (p.x - k.at<double>(0,2))/k.at<double>(0,0),
        (p.y - k.at<double>(1,2))/k.at<double>(1,1)
    );
}

void pose_2d2d(const vector<KeyPoint> &kp1, const vector<KeyPoint> &kp2, const vector<DMatch> &matches, const Mat &k, Mat &R, Mat &t){
    vector<Point2f> points1, points2;
    for(int i = 0; i < (int) matches.size(); i++){
        points1.push_back(kp1[matches[i].queryIdx].pt);
        points2.push_back(kp2[matches[i].trainIdx].pt);
    }

    // compute fundamental matrix 
    Mat F = findFundamentalMat(points1,points2,FM_8POINT);
    cout << "fundamental_matrix is " << endl << F << endl;

    // compute essential matrix
    Mat E = findEssentialMat(points1,points2,k,FM_8POINT);
    cout << "essential_matrix is " << endl << E << endl;

    // compute homography matrix
    Mat H = findHomography(points1,points2,RANSAC,3);
    cout << "homography_matrix is " << endl << H << endl;

    recoverPose(E,points1,points2,k,R,t);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}

int main(int argc, char **argv){
    if (argc != 3) {
        cout << "usage: pose_estimation_2d2d img1 img2" << endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread(argv[1], IMREAD_COLOR);
    Mat img_2 = imread(argv[2], IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // Intrinsics
    const Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    Mat R, t;
    // recover rotation and translation
    pose_2d2d(keypoints_1,keypoints_2,matches,K,R,t);

    for( DMatch m : matches){
        // Compute keypoints on unit surface
        Point2d pt1 = pix2cam(keypoints_1[m.queryIdx].pt, K);
        Point2d pt2 = pix2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y1 = (Mat_<double>(3,1) << pt1.x,pt1.y,1);
        Mat y2 = (Mat_<double>(3,1) << pt2.x,pt2.y,1);
        Mat d = y2.t()*vec2rot(t)*R*y1;
        cout << "epipolar constraint = " << d << endl;
    }

    return 0;
}