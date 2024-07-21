/**
 * @file chess.cpp
 * @brief 棋盘入口
 * @author Aurora
 * @date 2024-7-20
 */
#include <iostream>
#include "chess.h"

Complex::Complex() : realPart(0), imaginaryPart(0) {}
Complex::Complex(int x, int y) : realPart(x), imaginaryPart(y) {}

Complex Complex::operator+(const Complex &complex) const {
    return {this->realPart + complex.realPart, this->imaginaryPart + complex.imaginaryPart};
}
Complex Complex::operator-(const Complex &complex) const {
    return {this->realPart - complex.realPart, this->imaginaryPart - complex.imaginaryPart};
}

Chessboard::Chessboard() : goBoard({{}}) {}
Chessboard::Chessboard(std::array<std::array<int, 19>, 19> board) : goBoard(board) { update(); }


bool Chessboard::drop(int color, Complex position) {
    // 检查合理性
    if (this->goBoard[position.realPart][position.imaginaryPart] == 0 && position.imaginaryPart >=0 && position.realPart >=0) {
        this->goBoard[position.realPart][position.imaginaryPart] = color;
        update();
        return true;
    }
    return false;
}

bool Chessboard::clear(Complex position) {
    // 判断该操作是否合理
    if (this->goBoard[position.realPart][position.imaginaryPart] != 0) {
        // 处理该操作
        this->goBoard[position.realPart][position.imaginaryPart] = 0;
        update();
        return true;
    }
    throw std::exception();
    return false;
}

std::array<std::array<int, 19>,19> Chessboard::draw() {
    for (auto i : this->goBoard) {
        for (auto j : i) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
    return this->goBoard;
}

void Chessboard::update() {
    Complex position(0, 0);
    Complex i(1, 0), j(0, 1);
    int color = 0, colorUp = 0, colorDown = 0, colorLeft = 0, colorRight = 0;
    for (auto x : this->goBoard) {
        for (auto y : x) {
            color = getColor(position);
            // 判断是否为空
            if (color == 0) {
                position = position + i;
                continue;
            }

            colorUp = getColor(position + j);
            colorDown = getColor(position - j);
            colorLeft = getColor(position - i);
            colorRight = getColor(position + i);

            if (colorUp == 0 || colorDown == 0 || colorLeft == 0 || colorRight == 0)  {
                position = position + i;
                continue;
            }

            if (color != colorUp &&
                color != colorDown &&
                color != colorLeft &&
                color != colorRight) {
                // 死亡
                this->goBoard[position.realPart][position.imaginaryPart] = color + 2;
            }
            position = position + i;
        }
        position.realPart = 0;
        position = position + j;
    }

    // 将值为3 / 4的棋子清除
    Complex clearPoint(0, 0);
    for (auto m : this->goBoard) {
        for (auto n : m) {
            std::cout << clearPoint.realPart << ", " << clearPoint.imaginaryPart << std::endl;
            if (getColor(clearPoint) == 3 || getColor(clearPoint) == 4) {
                // 清除这颗棋子
                clear(clearPoint);
            }
            clearPoint = clearPoint + i;
        }
        clearPoint.realPart = 0;
        clearPoint = clearPoint + j;
    }
}

int Chessboard::getColor(Complex position) {
    if (position.realPart <= 18 && position.imaginaryPart <= 18 && position.realPart >= 0 && position.imaginaryPart >= 0) {
        return this->goBoard[position.realPart][position.imaginaryPart];
    }
    return -1;
}
