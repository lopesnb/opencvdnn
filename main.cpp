#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

int main() {
    // 1. モデルの読み込み (ONNX形式)
    auto net = cv::dnn::readNetFromONNX("best.onnx");
    
    // GPUが使える場合は高速化 (任意)
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::VideoCapture cap(0); // カメラ開始
    cv::Mat frame;

    while (cap.read(frame)) {
        // 2. 画像をAIが読める形式(640x640)に変換
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0/255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false);
        net.setInput(blob);

        // 3. 推論実行
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // 4. 後処理 (ここが昨日と違う重要な部分！)
        cv::Mat output = outputs[0];
        // YOLOv11の出力形式は [1, 5, 8400] (中心x, 中心y, 幅, 高さ, スコア)
        output = output.reshape(1, output.size[1]);
        cv::transpose(output, output); // [8400, 5] に変換

        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        float x_factor = frame.cols / 640.0;
        float y_factor = frame.rows / 640.0;

        for (int i = 0; i < output.rows; ++i) {
            float confidence = output.at<float>(i, 4); // 5番目がスコア
            if (confidence > 0.5) { // 信頼度50%以上
                float x = output.at<float>(i, 0);
                float y = output.at<float>(i, 1);
                float w = output.at<float>(i, 2);
                float h = output.at<float>(i, 3);

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }

        // 重なった枠を一つにまとめる(NMS処理)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        for (int idx : indices) {
            cv::rectangle(frame, boxes[idx], cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "Log: " + std::to_string(confidences[idx]), 
                        cv::Point(boxes[idx].x, boxes[idx].y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Log Detection", frame);
        if (cv::waitKey(1) >= 0) break;
    }
    return 0;
}