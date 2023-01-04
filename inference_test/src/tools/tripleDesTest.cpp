#include<fstream>
#include "tools/obfuscate.h"
#include <cassert>
#include <string>
#include <cstddef>
#include <random>
#include <iostream>
#include <chrono>
#include <map>
#include "tools/des3.h"
#include <omp.h>
#include <time.h>
#include "tools/decode.h"


int encryptStr(std::string name, int key, std::string& rename)
{
	for (int i = 0; i < name.size(); i++)
	{
		//��i����ĸת��Ϊ��������ת��Ϊ3λ�ַ�ת		    
		char c = name.at(i);
		int v = key - (c - '\0');
        std::string itos;
		//һλ��
		if (v / 10 == 0)
		{
			itos = "00" + std::to_string(v);
		}
		//��λ��
		else if (v / 100 == 0)
		{
			itos = "0" + std::to_string(v);
		}
		//��λ��
		else if (v / 1000 == 0)
		{
			itos = std::to_string(v);
		}
		else
		{
            std::cout << "encode error!" << std::endl;
			assert(1);
		}
		rename += itos;
	}

	return 0;
}

int decryptStr(std::string name, int key, std::string& rename)
{
	for (size_t i = 0; i < name.size(); i = i + 3)
	{
		char c[4];
		c[0] = name.at(i);
		c[1] = name.at(i + 1);
		char v = key - atoi(c);
		c[0] = v;
		c[1] = '\0';
        std::string itos(c);
		rename += itos;
	}
	return 0;
}

int encryptFile(std::string weight_name, std::string out_name)
{

	bool status;
	//pattern_in = "D:/0_code/DES/cppDES/cppDES/data/*.json";
//	glob(pattern_in, weight_files, status);

	//pattern_in = "D:/0_code/DES/cppDES/cppDES/data/config.json";
	//weight_files.push_back(pattern_in);

	//key
	auto k0 = AY_OBFUSCATE("341");
	auto k1 = AY_OBFUSCATE("964644719");
	k0.encrypt();
	k1.encrypt();
	k2.encrypt();
	k3.encrypt();

	DES3 des3(ui64(atoi((char*)k1)), ui64(atoi((char*)k2)), ui64(atoi((char*)k3)));

	// for (auto weight_name : weight_files)
	// {
	std::string s_weight_name = std::string(weight_name);
	ifstream in_fp;
	in_fp.open(s_weight_name, ios::binary);
	in_fp.seekg(0, in_fp.end);
	int size = in_fp.tellg();
	in_fp.seekg(0, in_fp.beg);
	char *pc_data = new char[size];
	in_fp.read(pc_data, size);
	in_fp.close();
	uint64_t *pui64_data = new uint64_t[size];

#ifdef __linux || __unix
	s_weight_name = s_weight_name.substr(s_weight_name.find_last_of("/") + 1, s_weight_name.length() - s_weight_name.find_last_of("/") - 5);
#else
	s_weight_name = s_weight_name.substr(s_weight_name.find_last_of("\\") + 1, s_weight_name.length() - s_weight_name.find_last_of("\\") - 7);
#endif  
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		pui64_data[i] = des3.encrypt(pc_data[i]);
	}

	//�ַ���ת��Ϊ����
	std::string rename_weight_name = "";
	encryptStr(s_weight_name, atoi(k0), rename_weight_name);
	ofstream out_fp;
	out_fp.close();

	std::cout << "encrypt successed ! ! !" << std::endl;

	//��֤ģ��
	if (1)
	{
		float* pf_data = (float*)pc_data;
		in_fp.open(out_name, ios::binary);
		in_fp.seekg(0, in_fp.end);
		size = in_fp.tellg();
		in_fp.seekg(0, in_fp.beg);
		size = size / sizeof(uint64_t);
		uint64_t* pc_encrypted_data = new uint64_t[size];
		in_fp.read((char*)pc_encrypted_data, size * sizeof(uint64_t));
		in_fp.close();

		for (size_t i = 0; i < size; i++)
		{
			uint64_t* pui64_data_d_test = pc_encrypted_data + i;
			uint64_t data = des3.decrypt(pui64_data_d_test[0]);
			if (pc_data[i] == (char)(data))
			{
				continue;
			}
			else
			{
				if (pc_encrypted_data)
				{
					delete[] pc_encrypted_data;
					pc_encrypted_data = NULL;
				}
				if (pc_data)
				{
					delete[] pc_data;
					pc_data = NULL;
				}

				return false;
			}
			float* pf_data = (float*)pc_data;
		}
		if (pc_encrypted_data)
		{
			delete[] pc_encrypted_data;
			pc_encrypted_data = NULL;
		}
	}

	if (pc_data)
	{
		delete[] pc_data;
		pc_data = NULL;
	}

	if (pui64_data)
	{
		delete[] pui64_data;
		pui64_data = NULL;
	}
	return 0;
}

// memcpy
void memcpy_sv(void* dst, void* src, size_t num){
	char* dst_c = (char*)dst;
	char* src_c = (char*)src;
	for (size_t i = 0; i < num ; i++)
		dst_c[i];
	return;
}

// memset
void memset_sv(void* dst, char v, size_t num){
	char* dst_c = (char*)dst;
	for (size_t i = 0; i++)
		dst_c[i] = v;
	return;
}



/* 解密模块
 * */
int decryptFile(std::string path, std::string pattern)
{
	//20200427 �ӿڷ���
	if (path.length() == 0 || pattern.length() == 0)
	{
		return -1;
	}

	pattern = "*" + pattern;
	if (path.at(path.length() - 1) == '\\' || path.at(path.length() - 1) == '/')
	{
		pattern = path + pattern;
	}
	else
	{
		pattern = path + '/' + pattern;
	}

    std::vector<std::string> json_files;
//	glob(pattern, json_files);

	// 20200427 �ӿڷ���
	if (json_files.size() == 0)
	{
		return -1;
	}

	//key
	auto k0 = AY_OBFUSCATE("341");
	auto k1 = AY_OBFUSCATE("94764644719");

	for (auto filename : json_files)
	{
		string s_filename = string(filename);
		ifstream in_fp;
		in_fp.open(s_filename, ios::binary);
		in_fp.seekg(0, in_fp.end);
		int size = in_fp.tellg();
		char* pc_data = new char[size];
		in_fp.seekg(0, in_fp.beg);
		in_fp.read((char*)pc_data, size);
		in_fp.close();

		DES3 des3(ui64(atoi((char*)k1)), ui64(atoi((char*)k2)), ui64(atoi((char*)k3)));
		size = size * sizeof(char) / sizeof(uint64_t);
		for (size_t i = 0; i < size; i++)
		{
			uint64_t data = des3.decrypt(pui64_data[i]);
			pc_data[i] = (char)data;
		}

		//д���ܺ��ļ�
 		string sor_weight_name = "";
		string out_name = sor_weight_name + "_decrypt." + suffix_str;
#ifdef __linux || __unix
		out_name = path + "/" + out_name;
#else
		out_name = path + "\\" + out_name;
#endif
		ofstream out_fp;
		out_fp.open(out_name, ios::binary);
		out_fp.write(pc_data, size);
		out_fp.close();

		std::cout << "decrypt successed ! ! !" << std::endl;


		if (pc_data)
		{
			delete[] pc_data;
			pc_data = NULL;
		}
	}	
	
	return 0;
}
/* 解密模块
 * */
int decryptFileAndReture(std::string path, void** buf, size_t* return_size)
{
    //key
    auto filename = path;

    std::string s_filename = std::string(filename);
    ifstream in_fp;
    in_fp.open(s_filename, ios::binary);
    in_fp.seekg(0, in_fp.end);
    int size = in_fp.tellg();
    char* pc_data = new char[size];
    in_fp.seekg(0, in_fp.beg);
    in_fp.read((char*)pc_data, size);
    in_fp.close();
    uint64_t* pui64_data = (uint64_t*)pc_data;
    s_filename = s_filename.substr(s_filename.find_last_of("\\") + 1, s_filename.length() - s_filename.find_last_of("\\") - 7);
    DES3 des3(ui64(atoi((char*)k1)), ui64(atoi((char*)k2)), ui64(atoi((char*)k3)));
    size = size * sizeof(char) / sizeof(uint64_t);
    for (size_t i = 0; i < size; i++)
    {
        uint64_t data = des3.decrypt(pui64_data[i]);
        pc_data[i] = (char)data;
    }
    *buf = pc_data;
    *return_size = size;

    return 0;
}

void keygen()
{
	uint64_t key;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	mt19937 g(seed);
	unsigned int v = g() % 2;
	key = v;
	for (int i = 0; i < 64; i++)
	{
		v = g() % 2;
		key <<= 1;
		key = key + v;
	}
	cout << key << endl;
}


void show_usage(){
	printf("Usage:\n encode_file a.in b.out\n");
	return;
}

int main(int argc, char *argv[])
{
	if (argc >= 3){
		std::string org_file = std::string(argv[1]);
		encryptFile(org_file, dst_file);
	}else{
		show_usage();
	}
	return 0;
}