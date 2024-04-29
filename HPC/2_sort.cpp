#include <bits/stdc++.h> // Include necessary libraries
#include <omp.h>


// Name : Harish Sugandhi , Saurabh Shete

using namespace std;

// Function to perform Bubble Sort
void bubbleSort(vector<int> &arr, int n)
{
    for (int i = 0; i < n - 1; i++) // Outer loop for passes
    {
        for (int j = 0; j < n - i - 1; j++) // Inner loop for comparisons
        {
            if (arr[j] > arr[j + 1]) // Swap if adjacent elements are in the wrong order
                swap(arr[j], arr[j + 1]);
        }
    }
}

// Parallel version of Bubble Sort
void parallelBubbleSort(vector<int> &arr, int n)
{
    for (int i = 0; i < n; i++) // Outer loop for passes
    {
#pragma omp parallel for // Parallel loop for odd indexed elements
        for (int j = 1; j < n; j += 2)
        {
            int thread_num = omp_get_thread_num(); // Get thread number
            cout << "\nThread " << thread_num << " is executing parallelBubbleSort() for odd indexed elements\n";
            if (arr[j - 1] > arr[j]) // Swap if adjacent elements are in the wrong order
                swap(arr[j - 1], arr[j]);
        }

#pragma omp barrier // Wait for all threads to finish before proceeding

#pragma omp parallel for // Parallel loop for even indexed elements
        for (int j = 2; j < n; j += 2)
        {
            int thread_num = omp_get_thread_num(); // Get thread number
            cout << "\nThread " << thread_num << " is executing parallelBubbleSort() for even indexed elements\n";
            if (arr[j - 1] > arr[j]) // Swap if adjacent elements are in the wrong order
                swap(arr[j - 1], arr[j]);
        }
    }
}

// Merge function for Merge Sort
void merge(vector<int> &arr, int low, int mid, int high)
{
    int n1 = mid - low + 1;
    int n2 = high - mid;

    int left[n1];
    int right[n2];

    for (int i = 0; i < n1; i++)
        left[i] = arr[low + i];

    for (int j = 0; j < n2; j++)
        right[j] = arr[mid + j + 1];

    int i = 0, j = 0, k = low;
    while (i < n1 && j < n2)
    {
        if (left[i] <= right[j])
        {
            arr[k] = left[i];
            i++;
        }
        else
        {
            arr[k] = right[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        arr[k++] = left[i++];
    }

    while (j < n2)
    {
        arr[k++] = right[j++];
    }
}

// Recursive function for Sequential Merge Sort
void mergeSort(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int mid = (low + high) / 2;

        mergeSort(arr, low, mid);
        mergeSort(arr, mid + 1, high);

        merge(arr, low, mid, high);
    }
}

// Parallel version of Merge Sort
void parallelMergeSort(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int mid = (low + high) / 2;

#pragma omp parallel sections // Divide the work into two sections
        {
#pragma omp section // First section for the left half
            {
                int thread_num = omp_get_thread_num(); // Get thread number
                cout << "\nThread " << thread_num << " is executing parallelMergeSort() for the left half\n";
                mergeSort(arr, low, mid);
            }

#pragma omp section // Second section for the right half
            {
                int thread_num = omp_get_thread_num(); // Get thread number
                cout << "\nThread " << thread_num << " is executing parallelMergeSort() for the right half\n";
                mergeSort(arr, mid + 1, high);
            }
        }

        merge(arr, low, mid, high); // Merge the two halves
    }
}

int main()
{
    int n;                       // Number of elements
    double start_time, end_time; // Variables to store time
    cout << "\nEnter no of elements:- ";
    cin >> n;                       // Input the number of elements
    vector<int> arr(n), arrCopy(n); // Create vectors to store the elements and a copy for sorting

    for (int i = 0; i < n; i++)
    {
        cout << "\nEnter element no " << i + 1 << ":- ";
        cin >> arr[i];       // Input the elements
        arrCopy[i] = arr[i]; // Copy the elements for sorting
    }

    int ch;   // Choice variable for menu
    while (1) // Menu loop
    {
        cout << "\n\n============ MENU ============\n\n";
        cout << "\n1.Bubble Sort";
        cout << "\n2.Merge Sort";
        cout << "\n0.Exit";
        cout << "\nEnter your choice:- ";
        cin >> ch; // Input choice

        switch (ch)
        {
        case 1:
            start_time = omp_get_wtime(); // Get start time
            bubbleSort(arr, n);           // Perform sequential Bubble Sort
            end_time = omp_get_wtime();   // Get end time

            cout << "\nSequential Bubble sort took " << end_time - start_time << " seconds"; // Print time taken
            for (auto i : arr)                                                               // Print sorted array
                cout << i << " ";
            cout << "\n";

            start_time = omp_get_wtime();   // Get start time
            parallelBubbleSort(arrCopy, n); // Perform parallel Bubble Sort
            end_time = omp_get_wtime();     // Get end time

            cout << "\nParallel Bubble sort took " << end_time - start_time << " seconds"; // Print time taken
            for (auto i : arrCopy)                                                         // Print sorted array
                cout << i << " ";
            cout << "\n";
            break;

        case 2:
            start_time = omp_get_wtime(); // Get start time
            mergeSort(arr, 0, n - 1);     // Perform sequential Merge Sort
            end_time = omp_get_wtime();   // Get end time

            cout << "\nSequential Merge sort took " << end_time - start_time << " seconds"; // Print time taken
            for (auto i : arr)                                                              // Print sorted array
                cout << i << " ";
            cout << "\n";

            start_time = omp_get_wtime();         // Get start time
            parallelMergeSort(arrCopy, 0, n - 1); // Perform parallel Merge Sort
            end_time = omp_get_wtime();           // Get end time

            cout << "\nParallel Merge sort took " << end_time - start_time << " seconds"; // Print time taken
            for (auto i : arrCopy)                                                        // Print sorted array
                cout << i << " ";
            cout << "\n";
            break;

        case 0:
            exit(0); // Exit program
            break;

        default:
            cout << "\nEnter correct choice\n"; // Print error message for invalid choice
            break;
        }
    }
}