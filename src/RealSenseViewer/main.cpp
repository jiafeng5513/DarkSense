#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <yolo_v2_class.hpp>
#include <vector>

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;

}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, rs2::depth_frame origin_depth,int current_det_fps = -1, int current_cap_fps = -1) {
    int const colors[6][3] = { {1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0} };

    for (auto& i : result_vec)
    {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        //画矩形框
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id)
        {
            std::string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            //画标题框
            auto leftUp = cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0));
            auto rightDown = cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1));
            cv::rectangle(mat_img, leftUp, rightDown, color, CV_FILLED, 8, 0);
            //算中心点距离
            auto center = cv::Point2f(i.x+ i.w/2, i.y+ i.h / 2);
            float dist_to_center = origin_depth.get_distance(center.x,center.y);//用来计算距离
            //画出中心点
            cv::circle(mat_img, center, 5, color, 5);
            //标题
            std::stringstream strStream;
            strStream << obj_name << " ";
            if (dist_to_center > 0) {
                strStream << dist_to_center;
            }
            else {
                strStream << "INF";
            }
            std::string msg = strStream.str();
            putText(mat_img, msg, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0)
    {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(
            current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}

void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
    for (auto& i : result_vec)
    {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

void run()
{
    std::string  names_file = "";
    std::string  cfg_file = "";
    std::string  weights_file = "";

    auto obj_names = objects_names_from_file(names_file);

    Detector detector(cfg_file, weights_file);

    cv::VideoCapture cam(0);

    while (true)
    {

       /* cv::Mat* mat_img = new cv::Mat();
        cam >> *mat_img;

        auto start = std::chrono::steady_clock::now();
        std::vector<bbox_t> result_vec = detector.detect(*mat_img);
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> spent = end - start;
        std::cout << " Time: " << spent.count() << " sec \n";

        draw_boxes(*mat_img, result_vec, obj_names);

        m_entity->setDetectionPlayer(*mat_img);

        show_console_result(result_vec, obj_names);*/
    }
}


int main(int argc, char* argv[]) try
{
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    using namespace cv;
    const auto obj_window_name = "Object Detection";
    const auto depth_window_name = "Depth Map";

    namedWindow(obj_window_name, WINDOW_AUTOSIZE);
    namedWindow(depth_window_name, WINDOW_AUTOSIZE);

    std::string  names_file = "E:/VisualStudio/DarkSense/src/yolov3/coco.names";
    std::string  cfg_file = "E:/VisualStudio/DarkSense/src/yolov3/yolov3.cfg";
    std::string  weights_file = "E:/VisualStudio/DarkSense/src/yolov3/yolov3.weights";
    auto obj_names = objects_names_from_file(names_file);
    Detector detector(cfg_file, weights_file);


    while (waitKey(1) < 0 && getWindowProperty(obj_window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        //从RS流中拿到数据
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        rs2::depth_frame origin_depth = data.get_depth_frame();
        rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
        rs2::frame rgb_frame = data.get_color_frame();

        // Query frame size (width and height)
        const int w = depth.as<rs2::video_frame>().get_width();
        const int h = depth.as<rs2::video_frame>().get_height();

        //深度图
        Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
        //显示深度图
        cv::imshow(depth_window_name, image);
        //RGB图像转为BGR
        Mat RGB_image(Size(w, h), CV_8UC3, (void*)rgb_frame.get_data(), Mat::AUTO_STEP);
        std::vector<Mat> channels;
        split(RGB_image, channels);//分割image1的通道
        Mat channels1 = channels[0];//R
        Mat channels2 = channels[1];//G
        Mat channels3 = channels[2];//B
        Mat BGR_image;
        merge(std::vector<Mat>{ channels3, channels2, channels1}, BGR_image);
        //目标检测
        std::vector<bbox_t> result_vec = detector.detect(BGR_image);
        //标框,类别,中心点距离
        draw_boxes(BGR_image, result_vec, obj_names, origin_depth);
        //显示目标检测结果
        cv::imshow(obj_window_name, BGR_image);
        //控制台输出
        show_console_result(result_vec, obj_names);
        
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}



