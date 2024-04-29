#include <bits/stdc++.h> // Include necessary libraries
#include <omp.h>

using namespace std;

/*
Name : Harish Sugandhi , Saurabh Shete
*/

// Function to find minimum value in the array
int min(vector<int> &arr, int n)
{
    int minVal = 1e6; // Initialize minVal to a large value

#pragma omp parallel for reduction(min : minVal) // OpenMP directive for parallel execution with reduction
    for (int i = 0; i < n; i++)                  // Loop through the array
    {
        // Get thread number
        int thread_num = omp_get_thread_num();
        // Print thread number and array element
        cout << "\nThread " << thread_num << " is executing min() and the value is " << arr[i] << "\n";
        if (minVal > arr[i]) // Update minVal if current element is smaller
            minVal = arr[i];
    }

    return minVal; // Return minimum value
}

// Function to find maximum value in the array
int max(vector<int> &arr, int n)
{
    int maxVal = 0; // Initialize maxVal to 0

#pragma omp parallel for reduction(max : maxVal) // OpenMP directive for parallel execution with reduction
    for (int i = 0; i < n; i++)                  // Loop through the array
    {
        // Get thread number
        int thread_num = omp_get_thread_num();
        // Print thread number and array element
        cout << "\nThread " << thread_num << " is executing max() and the value is " << arr[i] << "\n";
        if (maxVal < arr[i]) // Update maxVal if current element is larger
            maxVal = arr[i];
    }

    return maxVal; // Return maximum value
}

// Function to calculate sum of elements in the array
int sum(vector<int> &arr, int n)
{
    int sumVal = 0; // Initialize sumVal to 0

#pragma omp parallel for reduction(+ : sumVal) // OpenMP directive for parallel execution with reduction
    for (int i = 0; i < n; i++)                // Loop through the array
    {
        // Get thread number
        int thread_num = omp_get_thread_num();
        // Print thread number and array element
        cout << "\nThread " << thread_num << " is executing sum() and the value is " << arr[i] << "\n";
        sumVal += arr[i]; // Add current element to sum
    }

    return sumVal; // Return sum of elements
}

// Function to calculate average of elements in the array
double avg(vector<int> &arr, int n)
{
    return (double)sum(arr, n) / n; // Return average as sum divided by number of elements
}

int main()
{
    omp_set_num_threads(3); // Set number of threads to 3
    int n;
    cout << "\nEnter no of elements:- ";
    cin >> n; // Input number of elements

    vector<int> arr(n);         // Vector to store elements
    for (int i = 0; i < n; i++) // Loop to input elements
    {
        cout << "\nEnter element no " << i + 1 << ":- ";
        cin >> arr[i]; // Input element
    }

    int ch;
    double start_time, end_time;

    while (1) // Menu loop
    {
        cout << "\n\n=========== MENU =============\n\n";
        cout << "\n1.Min Value";
        cout << "\n2.Max Value";
        cout << "\n3.Sum Value";
        cout << "\n4.Avg Value";
        cout << "\n0.Exit";
        cout << "\nEnter your choice:- ";
        cin >> ch; // Input choice

        switch (ch)
        {
        case 1:                                                                // Minimum value
            start_time = omp_get_wtime();                                      // Start measuring time
            cout << "\nMinimum Value: " << min(arr, n);                        // Call min function and print result
            end_time = omp_get_wtime();                                        // End measuring time
            cout << "\nTime taken: " << end_time - start_time << " seconds\n"; // Print execution time
            break;

        case 2:                                                                // Maximum value
            start_time = omp_get_wtime();                                      // Start measuring time
            cout << "\nMaximum Value: " << max(arr, n);                        // Call max function and print result
            end_time = omp_get_wtime();                                        // End measuring time
            cout << "\nTime taken: " << end_time - start_time << " seconds\n"; // Print execution time
            break;

        case 3:                                                                // Sum of values
            start_time = omp_get_wtime();                                      // Start measuring time
            cout << "\nSum of Values: " << sum(arr, n);                        // Call sum function and print result
            end_time = omp_get_wtime();                                        // End measuring time
            cout << "\nTime taken: " << end_time - start_time << " seconds\n"; // Print execution time
            break;

        case 4:                                                                // Average value
            start_time = omp_get_wtime();                                      // Start measuring time
            cout << "\nAverage Value: " << avg(arr, n);                        // Call avg function and print result
            end_time = omp_get_wtime();                                        // End measuring time
            cout << "\nTime taken: " << end_time - start_time << " seconds\n"; // Print execution time
            break;

        case 0:      // Exit
            exit(0); // Exit the program
            break;

        default:
            cout << "\nEnter correct choice"; // Print error message for invalid choice
            break;
        }
    }

    return 0; // End of the program
}

/*
 --- g++ minmaxsumavg.cpp -o minmaxsumavg -fopenmp

 --- ./minmaxsumavg

Enter no of elements:- 5

Enter element no 1:- 3

Enter element no 2:- 4

Enter element no 3:- 5

Enter element no 4:- 1

Enter element no 5:- 2
*/