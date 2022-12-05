#include <iostream>
#include <omp.h>
#include <fstream>
#include <cstdlib>
using namespace std;
#include <sys/time.h>
#include <cctype>
#include <cmath>
#define MAX_SIZE_LINE 1000

#ifndef MAXIMUM_MEM_ALLOWED
#define MAXIMUM_MEM_ALLOWED 313000000
#endif // MAXIMUM_MEM_ALLOWED


/*

    -DMAXIMUM_MEM_ALLOWED=313000000

    g++ osm.cpp -O3 -fopenmp -o sasmo

    g++ osm_mem.cpp -O3 -fopenmp -o sasm268 -DMAXIMUM_MEM_ALLOWED=321000000
    g++ osm_mem.cpp -O3 -fopenmp -o sasm377 -DMAXIMUM_MEM_ALLOWED=500000000
    g++ osm_mem.cpp -O3 -fopenmp -o sasm538 -DMAXIMUM_MEM_ALLOWED=750000000
    g++ osm_mem.cpp -O3 -fopenmp -o sasm494 -DMAXIMUM_MEM_ALLOWED=800000000
    g++ osm_mem.cpp -O3 -fopenmp -o sasm198 -DMAXIMUM_MEM_ALLOWED=1014000000
*/
int lengthOfString1 = 18439, lengthOfString2 = 18439;
bool PRINT_CMAX = false;
bool PRINT_CMIN = false;
bool PRINT_DP = false;
bool PRINT_SEQS = false;
int FIND_BW_MAX_MIN = false;

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif // ESM_OMP_CPP

int** create2DArray(int m, int n)
{
    // Allocate memory for the matrix
    int** arr2D = new int* [m];

    int* memory = new int[m * n];

    for (int i = 0; i < m; i++)
    {
        arr2D[i] = &memory[i * n];
    }

    return arr2D;
}


bool hasEnding(char* name, const char* end);

const int ALPHASIzE = 5;
const unsigned char  AlphaBATES[] = "acgtn";



long findFuzzyStrings(char* str1, char* str2, int m, int n, int threshold)
{
    int* edC, * edP;
    edC = new int[n + 1];
    edP = new int[n + 1];
    int** Mi = create2DArray(ALPHASIzE, n + 1);
    // cudaMalloc((void**)&mF, mFlagsTable_BYTES);
    int* cmax, * cmin, * cmax_pre, * cmin_pre;

    cmax = new int[n + 1];
    cmin = new int[n + 1];
    cmax_pre = new int[n + 1];
    cmin_pre = new int[n + 1];


    int current_tasks = n + 1;

    #pragma omp parallel for
    for (int i = 0; i < ALPHASIzE; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            if (j == 0)
            {
                Mi[i][j] = 0;
            }
            else if (AlphaBATES[i] == str2[j - 1])
            {
                Mi[i][j] = j;
            }
            else
            {
                Mi[i][j] = Mi[i][j - 1];
            }
        }
    }


    //printEt2<<<1,1>>>(Mi,n);

    // cudaDeviceSynchronize();
    // int* dp = new int[n + 1];
    // int* cmax = new int[n + 1];
    // int* cmin = new int[n + 1];/ cudaDeviceSynchronize();


    #pragma omp parallel for
    for (int j = 0; j <= n; j++)
    {
        edP[j] = cmax_pre[j] = cmin_pre[j] = 0;
    }

    // #pragma omp 
    for (int i = 1; i <= m; i++)
    {
        unsigned char chS1 = str1[i - 1];
        int ci = 0;
        if (chS1 == AlphaBATES[1])
            ci = 1;
        else if (chS1 == AlphaBATES[2])
            ci = 2;
        else if (chS1 == AlphaBATES[3])
            ci = 3;
        else if (chS1 == AlphaBATES[4])
            ci = 4;

        edC[0] = i;
        cmax[0] = cmin[0] = 0;

        #pragma omp parallel for
        for (int j = 1; j <= n; j++)
        {
            bool mFlag = Mi[ci][j] == j;
            int diag = edP[j - 1];
            if (mFlag)
            {
                edC[j] = diag;
                cmax[j] = 1 + cmax_pre[j - 1];
                cmin[j] = 1 + cmin_pre[j - 1];
            }
            else
            {

                int y = edP[j] + 1;
                int z = diag + 1;
                int mi = Mi[ci][j];
                int diff_lm = j - mi;
                int x = mi > 0 ? edP[mi - 1] + diff_lm : y + z + 1;

                if (y <= x && y <= z) // delete
                {
                    edC[j] = y;
                    cmin[j] = cmin_pre[j];


                    if (x == y)
                    {
                        cmax[j] = cmax_pre[mi - 1] + diff_lm + 1;
                    }
                    else if (y == z)
                    {
                        cmax[j] = 1 + cmax_pre[j - 1];
                    }
                    else
                    {
                        cmax[j] = cmax_pre[j];
                    }
                }

                else if (z <= y && z <= x)   // replace
                {
                    edC[j] = z;
                    cmin[j] = 1 + cmin_pre[j - 1];

                    if (x == z)
                    {
                        cmax[j] = cmax_pre[mi - 1] + diff_lm;
                    }
                    else
                    {
                        cmax[j] = 1 + cmax_pre[j - 1];
                    }

                }
                else // if (x <= y && x <= z)
                {
                    edC[j] = x;
                    cmin[j] = cmin_pre[mi - 1] + diff_lm;
                    cmax[j] = cmax_pre[mi - 1] + diff_lm;
                }
            }
        }

        int* temp = edC;
        edC = edP;
        edP = temp;


        temp = cmax;
        cmax = cmax_pre;
        cmax_pre = temp;

        temp = cmin;
        cmin = cmin_pre;
        cmin_pre = temp;
    }


    int* temp = edC;
    edC = edP;
    edP = temp;


    temp = cmax;
    cmax = cmax_pre;
    cmax_pre = temp;

    temp = cmin;
    cmin = cmin_pre;
    cmin_pre = temp;


    //printEd<<<1,1>>>(ed,m,n);

    int* dp = edC;

    for (int i = 0; i < 1000; i++)
    {
        cout << dp[i] << "\t" << cmax[i] << "\t" << cmin[i] << endl;
    }


    long count_strings = 0;
    #pragma omp parallel for reduction(+: count_strings) num_threads(omp_get_max_threads())
    for (int i = 1; i <= n; i++)
    {
        if (dp[i] <= threshold)
        {

            if (cmin[i] > 0)
            {
                count_strings++;
                // cout.write(&str2[i - cmin[i]], cmin[i]);
                // cout << "(" << i - cmin[i] << "," << i - 1 << "," << dp[i] << ")";
                // cout << endl;
            }

            int idx = i - cmax[i];
            idx--;
            int tt = 1;
            while (dp[i] + tt <= threshold && idx >= 0)
            {

                count_strings++;
                // cout.write(&str2[idx], cmax[i] + tt);
                // cout << "(" << idx << "," << cmax[i] + tt + idx - 1 << "," << dp[i] + tt << ")";
                // cout << endl;
                idx--;
                tt++;
            }


            if (cmin[i] != cmax[i])
            {
                count_strings++;
                // cout.write(&str2[i - cmax[i]], cmax[i]);
                // cout << "(" << i - cmax[i] << "," << i - 1 << "," << dp[i] << ")";
                // cout << endl;

                if (FIND_BW_MAX_MIN)
                {
                    int s = i - cmax[i];
                    int e = (i - cmin[i]);
                    if (cmin[i] < m)
                    {
                        e = i - m;
                    }

                    if (s < e && s + 1 != e)
                    {
                        int oc_len = e - s - 1;
                        int* oc = new int[oc_len];

                        // if (i == n)
                        // {
                        //     cout << "yeah" << endl;
                        // }

                        for (int ii = s + 1, k = 0, j = 0; ii < e; ii++, k++)
                        {
                            if (str2[ii] == str1[0])
                            {
                                oc[k] = 1;
                                j++;
                            }
                            else if (ii == s + 1 || str2[ii] != str1[j])
                            {
                                oc[k] = 0;
                                j = 0;
                            }
                            else
                            {
                                oc[k] = oc[k - 1] + 1;
                                j++;
                            }

                            // if (i == n)
                            // {
                            //     cout << str2[ii] << ":" << oc[k] << ',';
                            // }
                        }

                        // cout << endl;

                        for (int j = oc_len - 1, k = 0; j >= 0; j--, k++)
                        {
                            if (oc[j] == 0)
                            {
                                break;
                            }
                            else if (oc[j] == 1)
                            {
                                int len = cmax[i] - oc_len + k;
                                count_strings++;
                                // cout.write(&str2[i - len], len);
                                // cout << "(" << i - len << "," << i - 1 << "," << dp[i] << ")";
                                // cout << endl;
                            }
                        }
                        delete[] oc;
                    }
                }
            }
        }
    }

    // cout << "Total strings found are: " << count_strings << endl;
    delete[] edC;
    delete[] edP;
    delete[] cmax;
    delete[] cmin;
    delete[] cmax_pre;
    delete[] cmin_pre;
    delete[] Mi[0];
    delete[] Mi;

    return count_strings;
}


int main(int argc, char** argv)
{

    if (argc < 4)
    {
        cout << "Please provide following arguments [file name pattern] [text file name] [error threshold percentage 50% def]" << endl;
        return 0;
    }

    omp_set_num_threads(omp_get_max_threads());

    char* fname1 = argv[1];
    char* fname2 = argv[2];
    int threshold = -1;

    bool isfasta1 = false;
    bool isfasta2 = false;

    if (hasEnding(fname1, ".fasta"))
    {
        isfasta1 = true;
    }

    if (hasEnding(fname2, ".fasta"))
    {
        isfasta2 = true;
    }

    ifstream fin1, fin2;
    fin1.open(fname1);


    if (!fin1.is_open())
    {
        cout << "File " << fname1 << " does not exist" << endl;
        return 0;
    }

    fin2.open(fname2);

    if (!fin2.is_open())
    {
        cout << "File " << fname2 << " does not exist" << endl;
        fin1.close();
        return 0;
    }
    char ch;

    lengthOfString1 = 0;

    if (isfasta1)
    {
        string line;
        getline(fin1, line);
    }
    while (fin1 >> ch)
    {
        if (ch != '\n')
        {
            lengthOfString1++;
        }
    }
    fin1.close();
    fin1.open(fname1);


    lengthOfString2 = 0;
    if (isfasta2)
    {
        string line;
        getline(fin2, line);
    }
    while (fin2 >> ch)
    {
        if (ch != '\n')
        {
            lengthOfString2++;
        }
    }
    fin2.close();
    fin2.open(fname2);


    if (argc > 3)
    {
        int tt = atoi(argv[3]);

        if (tt > 0 && tt < 101)
        {
            threshold = lengthOfString1 * (tt / 100.0);
        }
        else
        {
            threshold = lengthOfString1 / 2;
        }
    }

    if (threshold < 0)
    {
        threshold = lengthOfString1 / 2;
    }
    char* string1 = new char[lengthOfString1 + 1];
    char* string2 = new char[lengthOfString2 + 1];


    if (isfasta1)
    {
        string line;
        getline(fin1, line);
    }
    for (int iu = 0; iu < lengthOfString1; iu++)
    {
        fin1 >> ch;
        if (ch != '\n')
            string1[iu] = ch;
    }


    if (isfasta2)
    {
        string line;
        getline(fin2, line);
    }
    for (int iu = 0; iu < lengthOfString2; iu++)
    {
        fin2 >> ch;
        if (ch != '\n')
            string2[iu] = ch;
    }

    fin1.close();
    fin2.close();


    string1[lengthOfString1] = string2[lengthOfString2] = '\0';
    cout << "Size m = " << lengthOfString1 << " and n = " << lengthOfString2 << " and threshold = " << threshold << endl;

    if (PRINT_SEQS)
    {
        cout << "Pattern: " << string1 << endl;
        cout << "Text: " << string2 << endl;
    }

    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);;

    if (MAXIMUM_MEM_ALLOWED == -1)
    {
        long count_strings = findFuzzyStrings(string1, string2, lengthOfString1, lengthOfString2, threshold);
        cout << "Total strings found are: " << count_strings << endl;
    }
    else
    {
        unsigned long mem_needed = (lengthOfString2 + 1) * (unsigned long)160;

        cout << "Needed: " << mem_needed << " of string length " << lengthOfString2 + 1 << endl;

        if (mem_needed <= MAXIMUM_MEM_ALLOWED)
        {
            long count_strings = findFuzzyStrings(string1, string2, lengthOfString1, lengthOfString2, threshold);
            cout << "Total strings found are: " << count_strings << endl;
        }
        else
        {
            int max_n = MAXIMUM_MEM_ALLOWED / 160;

            cout << "Using partials = " << max_n << endl;

            int done = 0;
            long count_strings = 0;

            while (done < lengthOfString2)
            {
                int currrent_size = max_n;

                if (lengthOfString2 - done + 1 < max_n)
                {
                    currrent_size = lengthOfString2 - done + 1;
                }
                int start = done;

                if (done != 0)
                {
                    start -= (lengthOfString1 - 1);
                    currrent_size += (lengthOfString1 - 1);
                }
                count_strings += findFuzzyStrings(string1, &string2[start], lengthOfString1, currrent_size, threshold);
                done += max_n;
            }

            cout << "Total strings found are: " << count_strings << endl;
        }
    }

    gettimeofday(&tv2, NULL);
    long secs = tv2.tv_sec - tv1.tv_sec;
    long uSecs = tv2.tv_usec - tv1.tv_usec;
    double timeElapsed = secs + (uSecs / 1000000.0);
    printf("Time elapsed = %g s\n", timeElapsed);
    delete[] string1;
    delete[] string2;
    return 0;
}


bool hasEnding(char* name, const char* end)
{
    int len1 = 0;
    while (name[len1] != '\0')
    {
        len1++;
    }

    int len2 = 0;
    while (end[len2] != '\0')
    {
        len2++;
    }

    if (len2 > len1)
    {
        return false;
    }

    int i = len1 - len2;
    int j = 0;

    while (j != len2)
    {
        if (tolower(name[i]) != tolower(end[j]))
        {
            return false;
        }
        j++;
        i++;
    }

    return true;


}