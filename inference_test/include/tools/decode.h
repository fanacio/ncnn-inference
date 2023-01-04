//
// Created by Admin on 2022/3/24.
//

#ifndef MOBILENET_SHOOT_DECODE_H
#define MOBILENET_SHOOT_DECODE_H
/* 解密模块
 * */
#include<stdio.h>
#include<string>

int decryptFileAndReture(std::string path, void** buf, size_t* return_size);

#endif //MOBILENET_SHOOT_DECODE_H
