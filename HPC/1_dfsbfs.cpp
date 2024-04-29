
#include <bits/stdc++.h> // Include necessary libraries
#include <omp.h>

using namespace std;

// Name : Harish Sugandhi , Saurabh Shete
int V, E; // Global variables for number of vertices and edges

// Function to perform DFS traversal sequentially
void sequentialDfsUtil(int src, vector<bool> &vis, vector<int> adj[])
{
    vis[src] = true;
    cout << src << " "; // Print the current vertex

    for (int i = 0; i < adj[src].size(); i++)
    {
        int n = adj[src][i];
        if (!vis[n])
            sequentialDfsUtil(n, vis, adj); // Recursively call DFS for unvisited neighbors
    }
}

// Function to perform sequential DFS traversal
void sequentialDfs(int src, vector<int> adj[])
{
    vector<bool> vis(V, false);       // Initialize visited array
    sequentialDfsUtil(src, vis, adj); // Call utility function
}

// Function to perform parallel DFS traversal using OpenMP
void parallelDfsUtil(int src, vector<bool> &vis, vector<int> adj[])
{
    vis[src] = true;
    int thread_num = omp_get_thread_num(); // Get thread number
    cout << src << " (Thread " << thread_num << ") "
         << "\n"; // Print the current vertex and thread number

#pragma omp parallel for // OpenMP directive for parallel execution
    for (int i = 0; i < adj[src].size(); i++)
    {
        int n = adj[src][i];
        if (!vis[n])
            parallelDfsUtil(n, vis, adj); // Recursively call DFS for unvisited neighbors
    }
}

// Function to perform parallel DFS traversal using OpenMP
void parallelDfs(int src, vector<int> adj[])
{
    vector<bool> vis(V, false);     // Initialize visited array
    parallelDfsUtil(src, vis, adj); // Call utility function
}

// Function to perform sequential BFS traversal
void sequentialBfs(int src, vector<int> adj[])
{
    vector<bool> vis(V, false); // Initialize visited array
    queue<int> q;

    vis[src] = true;
    q.push(src);

    while (!q.empty())
    {
        int v = q.front();
        q.pop();

        cout << v << " "; // Print the current vertex
        for (int i = 0; i < adj[v].size(); i++)
        {
            int n = adj[v][i];
            if (!vis[n])
            {
                vis[n] = true;
                q.push(n); // Enqueue unvisited neighbors
            }
        }
    }
}

// Function to perform parallel BFS traversal using OpenMP
void parallelBfs(int src, vector<int> adj[])
{
    vector<bool> vis(V, false); // Initialize visited array
    queue<int> q;

    vis[src] = true;
    q.push(src);

    while (!q.empty())
    {
        int v = q.front();
        q.pop();

        int thread_num = omp_get_thread_num(); // Get thread number
        cout << v << " (Thread " << thread_num << ") "
             << "\n"; // Print the current vertex and thread number

#pragma omp parallel for // OpenMP directive for parallel execution
        for (int i = 0; i < adj[v].size(); i++)
        {
            int n = adj[v][i];
            if (!vis[n])
            {
                vis[n] = true;
                q.push(n); // Enqueue unvisited neighbors
            }
        }
    }
}

int main()
{
    cout << "\nEnter no of vertices:- ";
    cin >> V; // Input number of vertices
    cout << "\nEnter no of edges:- ";
    cin >> E;           // Input number of edges
    vector<int> adj[V]; // Adjacency list

    for (int i = 0; i < E; i++)
    {
        cout << "\nEnter edge no " << i + 1 << ":- ";
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v); // Add edge to adjacency list
        adj[v].push_back(u); // Add edge to adjacency list
    }

    int src;
    cout << "\nEnter src:- ";
    cin >> src; // Input source vertex

    // Measure sequential DFS performance
    cout << "\nSequential DFS:-\n";
    double start_time = omp_get_wtime();                                                  // Start measuring time
    sequentialDfs(src, adj);                                                              // Call sequential DFS
    double end_time = omp_get_wtime();                                                    // End measuring time
    cout << "\nSequential DFS Execution Time: " << end_time - start_time << " seconds\n"; // Print execution time

    // Measure parallel DFS performance
    cout << "\nParallel DFS:-\n";
    start_time = omp_get_wtime();                                                       // Start measuring time
    parallelDfs(src, adj);                                                              // Call parallel DFS
    end_time = omp_get_wtime();                                                         // End measuring time
    cout << "\nParallel DFS Execution Time: " << end_time - start_time << " seconds\n"; // Print execution time

    // Measure sequential BFS performance
    cout << "\nSequential BFS:-\n";
    start_time = omp_get_wtime();                                                         // Start measuring time
    sequentialBfs(src, adj);                                                              // Call sequential BFS
    end_time = omp_get_wtime();                                                           // End measuring time
    cout << "\nSequential BFS Execution Time: " << end_time - start_time << " seconds\n"; // Print execution time

    // Measure parallel BFS performance
    cout << "\nParallel BFS:-\n";
    start_time = omp_get_wtime();                                                       // Start measuring time
    parallelBfs(src, adj);                                                              // Call parallel BFS
    end_time = omp_get_wtime();                                                         // End measuring time
    cout << "\nParallel BFS Execution Time: " << end_time - start_time << " seconds\n"; // Print execution time

    return 0; // End of the program
}

// g++ 1_dfsbfs.cpp -o 1_dfsbfs -fopenmp
// ./ 1_dfsbfs
/*


Input and Output:-

Enter no of vertices:- 7

Enter no of edges:- 11

Enter edge no 1:- 0 1

Enter edge no 2:- 0 3

Enter edge no 3:- 1 3

Enter edge no 4:- 1 2

Enter edge no 5:- 1 5

Enter edge no 6:- 1 6

Enter edge no 7:- 3 4

Enter edge no 8:- 3 2

Enter edge no 9:- 2 4

Enter edge no 10:- 4 6
Enter edge no 11:- 2 5

Enter src:- 0

DFS:-
0 3 1 2 5 4 6
BFS:-
0 3 1 4 2 5 6  */

// Topic: Parallel Computing
/*
Parallel computing involves the simultaneous execution of multiple tasks or operations to achieve faster computation. It aims to divide a large problem into smaller sub-problems that can be solved concurrently, either by dividing data among multiple processors or by executing multiple threads simultaneously.iency in modern computing environments.

Parallel computing can lead to improved performance, scalability, and resource utilization in a wide range of applications, from scientific simulations and data analysis to web servers and multimedia processing. It encompasses various parallel programming models and techniques, including shared memory parallelism, distributed memory parallelism, and GPU computing.

However, parallel computing also presents challenges such as concurrency control, load balancing, and communication overhead, which require careful consideration during software design and implementation.

Overall, parallel computing plays a crucial role in addressing the increasing demand for computational power and efficiency in modern computing environments.
*/








// --------------// Another Code // ----------------------------------- /// 
/* 
#include <bits/stdc++.h> // Include necessary libraries
#include <omp.h>


// Name : Harish Sugandhi , Saurabh Shete


using namespace std;

int V, E; // Number of vertices and edges

// Function to perform DFS traversal sequentially
void sequentialDfsUtil(int src, vector<bool> &vis, vector<int> adj[])
{
    vis[src] = true;    // Mark the current vertex as visited
    cout << src << " "; // Print the current vertex

    for (int i = 0; i < adj[src].size(); i++) // Iterate through adjacent vertices
    {
        int n = adj[src][i];                // Get the adjacent vertex
        if (!vis[n])                        // If the adjacent vertex is not visited
            sequentialDfsUtil(n, vis, adj); // Recursively call DFS
    }
}

// Function to perform sequential DFS traversal
void sequentialDfs(int src, vector<int> adj[])
{
    vector<bool> vis(V, false);       // Initialize a boolean array to mark visited vertices
    sequentialDfsUtil(src, vis, adj); // Call the utility function
}

// Function to perform parallel DFS traversal using OpenMP
void parallelDfsUtil(int src, vector<bool> &vis, vector<int> adj[])
{
    vis[src] = true;    // Mark the current vertex as visited
    cout << src << " "; // Print the current vertex

#pragma omp parallel for // Parallel loop for exploring adjacent vertices
    for (int i = 0; i < adj[src].size(); i++)
    {
        int n = adj[src][i];              // Get the adjacent vertex
        if (!vis[n])                      // If the adjacent vertex is not visited
            parallelDfsUtil(n, vis, adj); // Recursively call DFS
    }
}

// Function to perform parallel DFS traversal using OpenMP
void parallelDfs(int src, vector<int> adj[])
{
    vector<bool> vis(V, false);     // Initialize a boolean array to mark visited vertices
    parallelDfsUtil(src, vis, adj); // Call the utility function
}

// Function to perform sequential BFS traversal
void sequentialBfs(int src, vector<int> adj[])
{
    vector<bool> vis(V, false); // Initialize a boolean array to mark visited vertices
    queue<int> q;               // Create a queue for BFS traversal

    vis[src] = true; // Mark the source vertex as visited
    q.push(src);     // Push the source vertex into the queue

    while (!q.empty()) // While the queue is not empty
    {
        int v = q.front(); // Get the front of the queue
        q.pop();           // Remove the front vertex from the queue

        cout << v << " "; // Print the current vertex

        for (int i = 0; i < adj[v].size(); i++) // Iterate through adjacent vertices
        {
            int n = adj[v][i]; // Get the adjacent vertex
            if (!vis[n])       // If the adjacent vertex is not visited
            {
                vis[n] = true; // Mark it as visited
                q.push(n);     // Push it into the queue
            }
        }
    }
}

// Function to perform parallel BFS traversal using OpenMP
void parallelBfs(int src, vector<int> adj[])
{
    vector<bool> vis(V, false); // Initialize a boolean array to mark visited vertices
    queue<int> q;               // Create a queue for BFS traversal

    vis[src] = true; // Mark the source vertex as visited
    q.push(src);     // Push the source vertex into the queue

    while (!q.empty()) // While the queue is not empty
    {
        int v = q.front(); // Get the front of the queue
        q.pop();           // Remove the front vertex from the queue

        cout << v << " "; // Print the current vertex

#pragma omp parallel for // Parallel loop for exploring adjacent vertices
        for (int i = 0; i < adj[v].size(); i++)
        {
            int n = adj[v][i]; // Get the adjacent vertex
            if (!vis[n])       // If the adjacent vertex is not visited
            {
                vis[n] = true; // Mark it as visited
                q.push(n);     // Push it into the queue
            }
        }
    }
}

int main()
{
    cout << "\nEnter no of vertices:- ";
    cin >> V; // Input number of vertices
    cout << "\nEnter no of edges:- ";
    cin >> E;           // Input number of edges
    vector<int> adj[V]; // Adjacency list representation of the graph

    for (int i = 0; i < E; i++) // Input edges of the graph
    {
        cout << "\nEnter edge no " << i + 1 << ":- ";
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v); // Add v to the adjacency list of u
        adj[v].push_back(u); // Add u to the adjacency list of v
    }

    int src; // Source vertex for traversal
    cout << "\nEnter src:- ";
    cin >> src;

    // Measure sequential DFS performance
    cout << "\nSequential DFS:-\n";
    double start_time = omp_get_wtime();
    sequentialDfs(src, adj);
    double end_time = omp_get_wtime();
    cout << "\nSequential DFS Execution Time: " << end_time - start_time << " seconds\n";

    // Measure parallel DFS performance
    cout << "\nParallel DFS:-\n";
    start_time = omp_get_wtime();
    parallelDfs(src, adj);
    end_time = omp_get_wtime();
    cout << "\nParallel DFS Execution Time: " << end_time - start_time << " seconds\n";

    // Measure sequential BFS performance
    cout << "\nSequential BFS:-\n";
    start_time = omp_get_wtime();
    sequentialBfs(src, adj);
    end_time = omp_get_wtime();
    cout << "\nSequential BFS Execution Time: " << end_time - start_time << " seconds\n";

    // Measure parallel BFS performance
    cout << "\nParallel BFS:-\n";
    start_time = omp_get_wtime();
    parallelBfs(src, adj);
    end_time = omp_get_wtime();
    cout << "\nParallel BFS Execution Time: " << end_time - start_time << " seconds\n";

    return 0; // End of the program
}  */