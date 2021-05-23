{
 "cells": [
  {
   "cell_type": "raw",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# #Task no 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def function1():\n",
    "    #create 2d array from 1,12 range\n",
    "    #dimention should be 6 rows, 2 columns\n",
    "    #and assign this array values in x variable\n",
    "    #Hint: you can use arange and reshape methods\n",
    "    x = np.arange(1,13).reshape((6,2))\n",
    "    \n",
    "    return x\n",
    "    \"\"\"\n",
    "    expected output:\n",
    "    [[ 1  2]\n",
    "    [ 3  4]\n",
    "    [ 5  6]\n",
    "    [ 7  8]\n",
    "    [ 9 10]\n",
    "    [11 12]]\n",
    "    \"\"\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2],\n",
       "       [ 3,  4],\n",
       "       [ 5,  6],\n",
       "       [ 7,  8],\n",
       "       [ 9, 10],\n",
       "       [11, 12]])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function1()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Task2\n",
    "def function2():\n",
    "    #create 3D array (3,3,3)\n",
    "    #must data type should have float64\n",
    "    #array value should be satart from 10 and end with 36 (both included)\n",
    "    # Hint: dtype, reshape \n",
    "    \n",
    "    x = np.arange(10,37,dtype=np.float64).reshape((3,3,3))     #wrtie your code here\n",
    "\n",
    "\n",
    "    return x\n",
    "    \"\"\"\n",
    "    Expected: out put\n",
    "array([[[10., 11., 12.],\n",
    "        [13., 14., 15.],\n",
    "        [16., 17., 18.]],\n",
    "       [[19., 20., 21.],\n",
    "        [22., 23., 24.],\n",
    "        [25., 26., 27.]],\n",
    "       [[28., 29., 30.],\n",
    "        [31., 32., 33.],\n",
    "        [34., 35., 36.]]])    \n",
    "    \"\"\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[10., 11., 12.],\n",
       "        [13., 14., 15.],\n",
       "        [16., 17., 18.]],\n",
       "\n",
       "       [[19., 20., 21.],\n",
       "        [22., 23., 24.],\n",
       "        [25., 26., 27.]],\n",
       "\n",
       "       [[28., 29., 30.],\n",
       "        [31., 32., 33.],\n",
       "        [34., 35., 36.]]])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function2()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 795,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task3\n",
    "def function3():\n",
    "    #extract those numbers from given array. those must exist in multiplication tables of 5,7\n",
    "    #example [35,70,105,..]\n",
    "    a = np.arange(1, 100*10+1).reshape((100,10))\n",
    "    x = a[(a%5 == 0)& (a%7 == 0)]#write your code here\n",
    "    return x\n",
    "    \"\"\"\n",
    "    Expected Output:\n",
    "     [35,  70, 105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455,\n",
    "       490, 525, 560, 595, 630, 665, 700, 735, 770, 805, 840, 875, 910,\n",
    "       945, 980] \n",
    "    \"\"\" "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 796,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 35,  70, 105, 140, 175, 210, 245, 280, 315, 350, 385, 420, 455,\n",
       "       490, 525, 560, 595, 630, 665, 700, 735, 770, 805, 840, 875, 910,\n",
       "       945, 980])"
      ]
     },
     "execution_count": 796,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function3()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 312,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task4\n",
    "def function4():\n",
    "    #Swap columns 1 and 2 in the array arr.\n",
    "\n",
    "    arr = np.arange(9).reshape(3,3)\n",
    "\n",
    "    return arr[:,[1, 0, 2]] #wrtie your code here\n",
    "    \"\"\"\n",
    "    Expected Output:\n",
    "          array([[1, 0, 2],\n",
    "                [4, 3, 5],\n",
    "                [7, 6, 8]])\n",
    "    \"\"\" "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 313,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 0, 2],\n",
       "       [4, 3, 5],\n",
       "       [7, 6, 8]])"
      ]
     },
     "execution_count": 313,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function4()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 328,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task5\n",
    "def function5():\n",
    "    #Create a null vector of size 20 with 4 rows and 5 columns with numpy function\n",
    "\n",
    "    z = np.zeros(20).reshape(4,5)#wrtie your code here\n",
    "\n",
    "    return z\n",
    "    \"\"\"\n",
    "    Expected Output:\n",
    "          array([[0, 0, 0, 0, 0],\n",
    "                [0, 0, 0, 0, 0],\n",
    "                [0, 0, 0, 0, 0],\n",
    "                [0, 0, 0, 0, 0]])\n",
    "    \"\"\" \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 329,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0., 0.],\n",
       "       [0., 0., 0., 0., 0.]])"
      ]
     },
     "execution_count": 329,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function5()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 336,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task6\n",
    "def function6():\n",
    "    # Create a null vector of size 10 but the fifth and eighth value which is 10,20 respectively\n",
    "\n",
    "    arr = np.zeros(10)#wrtie your code here\n",
    "    arr[4] = 10\n",
    "    arr[7] = 20\n",
    "    return arr\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 337,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.,  0.,  0.,  0., 10.,  0.,  0., 20.,  0.,  0.])"
      ]
     },
     "execution_count": 337,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function6()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 339,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task7\n",
    "def function7():\n",
    "    #  Create an array of zeros with the same shape and type as X. Dont use reshape method\n",
    "    x = np.arange(4, dtype=np.int64)\n",
    "\n",
    "    return np.zeros(4, dtype=np.int64)#write your code here\n",
    "\n",
    "    \"\"\"\n",
    "    Expected Output:\n",
    "          array([0, 0, 0, 0], dtype=int64)\n",
    "    \"\"\" "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 340,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 340,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function7()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 791,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task8\n",
    "def function8():\n",
    "    # Create a new array of 2x5 uints, filled with 6.\n",
    "\n",
    "    x = np.ones(10).reshape(2,5)*6, #write your code here\n",
    "\n",
    "    return x\n",
    "\n",
    "    \"\"\"\n",
    "     Expected Output:\n",
    "              array([[6, 6, 6, 6, 6],\n",
    "                     [6, 6, 6, 6, 6]], dtype=uint32)\n",
    "    \"\"\" "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 792,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([[6., 6., 6., 6., 6.],\n",
       "        [6., 6., 6., 6., 6.]]),)"
      ]
     },
     "execution_count": 792,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function8()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 385,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task9\n",
    "def function9():\n",
    "    # Create an array of 2, 4, 6, 8, ..., 100.\n",
    "\n",
    "    a = np.arange(0,101,2)# write your code here\n",
    "\n",
    "    return a\n",
    "\n",
    "    \"\"\"\n",
    "     Expected Output:\n",
    "              array([  2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,\n",
    "                    28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,\n",
    "                    54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,\n",
    "                    80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100])\n",
    "    \"\"\" "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 386,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,\n",
       "        26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,\n",
       "        52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,\n",
       "        78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100])"
      ]
     },
     "execution_count": 386,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function9()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 387,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task10\n",
    "def function10():\n",
    "    # Subtract the 1d array brr from the 2d array arr, such that each item of brr subtracts from respective row of arr.\n",
    "\n",
    "    arr = np.array([[3,3,3],[4,4,4],[5,5,5]])\n",
    "    brr = np.array([1,2,3])\n",
    "    subt = arr-brr[:,None]# write your code here \n",
    "\n",
    "    return subt\n",
    "\n",
    "    \"\"\"\n",
    "     Expected Output:\n",
    "               array([[2 2 2]\n",
    "                      [2 2 2]\n",
    "                      [2 2 2]])\n",
    "    \"\"\" \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 388,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2, 2, 2],\n",
       "       [2, 2, 2],\n",
       "       [2, 2, 2]])"
      ]
     },
     "execution_count": 388,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function10()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 393,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task11\n",
    "def function11():\n",
    "    # Replace all odd numbers in arr with -1 without changing arr.\n",
    "\n",
    "    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n",
    "    ans = np.where(arr%2, -1, arr)#write your code here \n",
    "\n",
    "    return ans\n",
    "\n",
    "    \"\"\"\n",
    "     Expected Output:\n",
    "              array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])\n",
    "    \"\"\" \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 394,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])"
      ]
     },
     "execution_count": 394,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function11()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 785,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task12\n",
    "def function12():\n",
    "    # Create the following pattern without hardcoding. Use only numpy functions and the below input array arr.\n",
    "    # HINT: use stacking concept\n",
    "\n",
    "    arr = np.array([1,2,3])\n",
    "    ans = np.hstack((np.repeat(arr, 3),arr, arr, arr)) #write your code here np.repeat(arr, 3), np.tile(arr, 3)\n",
    "     \n",
    "    return ans\n",
    "\n",
    "    \"\"\"\n",
    "     Expected Output:\n",
    "              array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])\n",
    "    \"\"\" \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 786,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])"
      ]
     },
     "execution_count": 786,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function12()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 745,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task13\n",
    "def function13():\n",
    "    # Set a condition which gets all items between 5 and 10 from arr.\n",
    "\n",
    "\n",
    "    arr = np.array([2, 6, 1, 9, 10, 3, 27])\n",
    "    ans = arr[(arr > 5) & (arr<10 )]#write your code here \n",
    "\n",
    "    return ans\n",
    "\n",
    "    \"\"\"\n",
    "     Expected Output:\n",
    "              array([6, 9])\n",
    "    \"\"\" "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 746,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([6, 9])"
      ]
     },
     "execution_count": 746,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function13()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 650,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task14\n",
    "def function14():\n",
    "    # Create an 8X3 integer array from a range between 10 to 34 such that the difference between each element is 1 and then Split the array into four equal-sized sub-arrays.\n",
    "    # Hint use split method\n",
    "\n",
    "\n",
    "    arr = np.arange(10, 34, 1).reshape(8,3) #write reshape code\n",
    "    ans = np.split(arr, 4)#write your code here \n",
    "\n",
    "    return ans\n",
    "\n",
    "    \"\"\"\n",
    "     Expected Output:\n",
    "       [array([[10, 11, 12],[13, 14, 15]]), \n",
    "        array([[16, 17, 18],[19, 20, 21]]), \n",
    "        array([[22, 23, 24],[25, 26, 27]]), \n",
    "        array([[28, 29, 30],[31, 32, 33]])]\n",
    "    \"\"\" \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 651,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[array([[10, 11, 12],\n",
       "        [13, 14, 15]]),\n",
       " array([[16, 17, 18],\n",
       "        [19, 20, 21]]),\n",
       " array([[22, 23, 24],\n",
       "        [25, 26, 27]]),\n",
       " array([[28, 29, 30],\n",
       "        [31, 32, 33]])]"
      ]
     },
     "execution_count": 651,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function14()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 696,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task15\n",
    "def function15():\n",
    "    #Sort following NumPy array by the second column\n",
    "\n",
    "\n",
    "    arr = np.array([[ 8,  2, -2],[-4,  1,  7],[ 6,  3,  9]])\n",
    "    ans = arr[np.argsort(arr[:, 1])]#write your code here \n",
    "\n",
    "    return ans\n",
    "\n",
    "    \"\"\"\n",
    "     Expected Output:\n",
    "           array([[-4,  1,  7],\n",
    "                   [ 8,  2, -2],\n",
    "                   [ 6,  3,  9]])\n",
    "    \"\"\" \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 697,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-4,  1,  7],\n",
       "       [ 8,  2, -2],\n",
       "       [ 6,  3,  9]])"
      ]
     },
     "execution_count": 697,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function15()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 699,
   "metadata": {},
   "outputs": [],
   "source": [
    "#task16\n",
    "def function16():\n",
    "    #Write a NumPy program to join a sequence of arrays along depth.\n",
    "\n",
    "\n",
    "    x = np.array([[1], [2], [3]])\n",
    "    y = np.array([[2], [3], [4]])\n",
    "    ans = np.hstack([x,y])#write your code here \n",
    "\n",
    "    return ans\n",
    "\n",
    "    \"\"\"\n",
    "     Expected Output:\n",
    "                [[[1 2]]\n",
    "                 [[2 3]]\n",
    "                 [[3 4]]]\n",
    "    \"\"\" \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 718,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2],\n",
       "       [2, 3],\n",
       "       [3, 4]])"
      ]
     },
     "execution_count": 718,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function16()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 725,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Task17\n",
    "def function17():\n",
    "    # replace numbers with \"YES\" if it divided by 3 and 5\n",
    "    # otherwise it will be replaced with \"NO\"\n",
    "    # Hint: np.where\n",
    "    arr = np.arange(1,10*10+1).reshape((10,10))\n",
    "    return np.where((arr%3 == 0) & (arr%5 == 0),\"YES\", \"NO\")          # Write Your Code HERE\n",
    "\n",
    "    #Excpected Out\n",
    "    \"\"\"\n",
    "     array([['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
    "       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
    "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],\n",
    "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
    "       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
    "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],\n",
    "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
    "       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
    "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],\n",
    "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO']],\n",
    "      dtype='<U3')\n",
    "    \"\"\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 726,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],\n",
       "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],\n",
       "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "       ['NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO'],\n",
       "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES'],\n",
       "       ['NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO']],\n",
       "      dtype='<U3')"
      ]
     },
     "execution_count": 726,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function17()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 735,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Task18\n",
    "def function18():\n",
    "    # count values of \"students\" are exist in \"piaic\"\n",
    "    piaic = np.arange(100)\n",
    "    students = np.array([5,20,50,200,301,7001])\n",
    "    x = np.intersect1d(piaic,students) # Write you code Here\n",
    "    x = len(x)\n",
    "    \n",
    "    return x\n",
    "\n",
    "    #Expected output: 3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 736,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 736,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function18()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 739,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Task19\n",
    "def function19():\n",
    "    #Create variable \"X\" from 1,25 (both are included) range values\n",
    "    #Convert \"X\" variable dimension into 5 rows and 5 columns\n",
    "    #Create one more variable \"W\" copy of \"X\" \n",
    "    #Swap \"W\" row and column axis (like transpose)\n",
    "    # then create variable \"b\" with value equal to 5\n",
    "    # Now return output as \"(X*W)+b:\n",
    "\n",
    "    X = np.arange(1,26).reshape(5,5)  # Write your code here\n",
    "    W = X.copy()  # Write your code here\n",
    "    W = W.T\n",
    "    b = 5  # Write your code here\n",
    "    output =  (X*W)+b  # Write your code here\n",
    "    return output\n",
    "\n",
    "    #expected output\n",
    "    \"\"\"\n",
    "    array([[  6,  17,  38,  69, 110],\n",
    "       [ 17,  54, 101, 158, 225],\n",
    "       [ 38, 101, 174, 257, 350],\n",
    "       [ 69, 158, 257, 366, 485],\n",
    "       [110, 225, 350, 485, 630]])\n",
    "    \"\"\"\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 740,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[  6,  17,  38,  69, 110],\n",
       "       [ 17,  54, 101, 158, 225],\n",
       "       [ 38, 101, 174, 257, 350],\n",
       "       [ 69, 158, 257, 366, 485],\n",
       "       [110, 225, 350, 485, 630]])"
      ]
     },
     "execution_count": 740,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function19()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 741,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Task20\n",
    "def function20():\n",
    "    #apply function \"abc\" on each value of Array \"X\"\n",
    "    x = np.arange(1,11)\n",
    "    def xyz(x):\n",
    "        return x*2+3-2\n",
    "\n",
    "    return xyz(x)#Write your Code here\n",
    "#Expected Output: array([ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 742,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21])"
      ]
     },
     "execution_count": 742,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function20()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
