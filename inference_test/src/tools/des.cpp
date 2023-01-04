#include "tools/des.h"
#include "tools/des_key.h"
#include "tools/des_data.h"
//#include "des_lookup.h"

//#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")

DES::DES(ui64 key)
{
    keygen(key);
}

ui64 DES::encrypt(ui64 block)
{
    return des(block, false);
}

ui64 DES::decrypt(ui64 block)
{
    return des(block, true);
}

ui64 DES::encrypt(ui64 block, ui64 key)
{
    DES des(key);
    return des.des(block, false);
}

ui64 DES::decrypt(ui64 block, ui64 key)
{
    DES des(key);
    return des.des(block, true);
}

void DES::keygen(ui64 key)
{
    // initial key schedule calculation
    ui64 permuted_choice_1 = 0; // 56 bits
    for (ui8 i = 0; i < 56; i++)
    {
        permuted_choice_1 <<= 1;
        permuted_choice_1 |= (key >> (64-PC1[i])) & LB64_MASK;
    }

    // Calculation of the 16 keys
    for (ui8 i = 0; i < 16; i++)
    {
        // key schedule, shifting Ci and Di

        ui64 permuted_choice_2 = (((ui64) C) << 28) | (ui64) D;

        sub_key[i] = 0; // 48 bits (2*24)
        for (ui8 j = 0; j < 48; j++)
        {
            sub_key[i] <<= 1;
            sub_key[i] |= (permuted_choice_2 >> (56-PC2[j])) & LB64_MASK;
        }
    }
}

ui64 DES::des(ui64 block, bool mode)
{
    // applying initial permutation
    block = ip(block);

    // dividing T' into two 32-bit parts
    ui32 L = (ui32) (block >> 32) & L64_MASK;
    ui32 R = (ui32) (block & L64_MASK);

    // 16 rounds
    for (ui8 i = 0; i < 16; i++)
    {
        ui32 F = mode ? f(R, sub_key[15-i]) : f(R, sub_key[i]);
        feistel(L, R, F);
    }


    return fp(block);
}

ui64 DES::ip(ui64 block)
{
    // initial permutation
    ui64 result = 0;
    for (ui8 i = 0; i < 64; i++)
    {
        result <<= 1;
        result |= (block >> (64-IP[i])) & LB64_MASK;
    }
    return result;
}



void DES::feistel(ui32 &L, ui32 &R, ui32 F)
{
    ui32 temp = R;
    R = L ^ F;
    L = temp;
}



//#pragma GCC pop_options
