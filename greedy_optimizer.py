import json
from typing import List, Dict, Tuple, Any
import time

def load_scenario(filename: str) -> Any:
    """Load a delivery scenario from a JSON file."""
    with open(f'scenarios/{filename}', 'r') as f:
        return json.load(f)

# ============================================
# PART 1: PACKAGE PRIORITIZATION (Activity Selection)
# ============================================

def maximize_deliveries(time_windows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Schedule the maximum number of non-overlapping deliveries.
    
    Args:
        time_windows: List of deliveries with 'start' and 'end' times
                     Example: [{'start': 9.0, 'end': 10.5}, {'start': 10.0, 'end': 11.0}]
    
    Returns:
        List of scheduled deliveries (non-overlapping, maximizing count)
    """
    if not time_windows:
        return []
    
    # Sort deliveries by their end time (greedy choice: earliest finish time)
    sorted_deliveries = sorted(time_windows, key=lambda x: x['end'])
    
    scheduled = []
    last_end_time = float('-inf')
    
    for delivery in sorted_deliveries:
        start = delivery['start']
        end = delivery['end']
        
        # If this delivery starts after or when the last one ends, schedule it
        if start >= last_end_time:
            scheduled.append(delivery)
            last_end_time = end
    
    return scheduled


# ============================================
# PART 2: TRUCK LOADING (Fractional Knapsack)
# ============================================

def optimize_truck_load(packages: List[Dict[str, float]], weight_limit: float) -> Dict[str, Any]:
    """
    Maximize total priority value of packages loaded within weight constraint.
    Can take fractions of packages.
    
    Args:
        packages: List of packages with 'weight' and 'value' (priority)
                 Example: [{'weight': 10.5, 'value': 100}, {'weight': 5.0, 'value': 30}]
        weight_limit: Maximum weight the truck can carry
    
    Returns:
        Dictionary with:
            - 'selected': List of packages with actual weight taken (may be fractional)
            - 'total_value': Sum of values achieved
    """
    if not packages or weight_limit <= 0:
        return {'selected': [], 'total_value': 0.0}
    
    # Calculate value-to-weight ratio for each package
    packages_with_ratio = []
    for p in packages:
        ratio = p['value'] / p['weight'] if p['weight'] > 0 else float('inf')
        packages_with_ratio.append({
            'weight': p['weight'],
            'value': p['value'],
            'ratio': ratio,
            'original_index': len(packages_with_ratio)  # For tracking
        })
    
    # Sort by value-to-weight ratio in descending order (greedy choice)
    sorted_packages = sorted(packages_with_ratio, key=lambda x: x['ratio'], reverse=True)
    
    selected = []
    remaining_weight = weight_limit
    total_value = 0.0
    
    for package in sorted_packages:
        if remaining_weight <= 0:
            break
            
        if package['weight'] <= remaining_weight:
            # Take the whole package
            selected.append({
                'weight': package['weight'],
                'value': package['value'],
                'fraction': 1.0
            })
            total_value += package['value']
            remaining_weight -= package['weight']
        else:
            # Take a fraction of the package
            fraction = remaining_weight / package['weight']
            selected.append({
                'weight': remaining_weight,
                'value': package['value'] * fraction,
                'fraction': fraction
            })
            total_value += package['value'] * fraction
            remaining_weight = 0
    
    return {
        'selected': selected,
        'total_value': total_value
    }


# ============================================
# PART 3: DRIVER ASSIGNMENT (Interval Partitioning)
# ============================================

def minimize_drivers(deliveries: List[Dict[str, float]]) -> Dict[int, List[Dict[str, float]]]:
    """
    Assign deliveries to the minimum number of drivers needed.
    
    Args:
        deliveries: List of deliveries with 'start' and 'end' times
                   Example: [{'start': 9.0, 'end': 10.5}, {'start': 10.0, 'end': 11.0}]
    
    Returns:
        Dictionary mapping driver IDs to their assigned deliveries
        Example: {0: [delivery1, delivery2], 1: [delivery3]}
    """
    if not deliveries:
        return {}
    
    # Sort deliveries by start time
    sorted_deliveries = sorted(deliveries, key=lambda x: x['start'])
    
    # Track end times for each driver's last delivery
    driver_end_times = []  # List of (driver_id, end_time)
    assignments = {}  # driver_id -> list of deliveries
    
    for delivery in sorted_deliveries:
        start = delivery['start']
        end = delivery['end']
        
        # Find a driver who is free at this start time
        assigned = False
        
        # Check existing drivers
        for i, (driver_id, driver_end) in enumerate(driver_end_times):
            if start >= driver_end:
                # This driver is free, assign to them
                driver_end_times[i] = (driver_id, end)
                assignments[driver_id].append(delivery)
                assigned = True
                break
        
        if not assigned:
            # Need a new driver
            new_driver_id = len(driver_end_times)
            driver_end_times.append((new_driver_id, end))
            assignments[new_driver_id] = [delivery]
    
    return assignments


import heapq

def minimize_drivers_heap(deliveries: List[Dict[str, float]]) -> Dict[int, List[Dict[str, float]]]:
    """
    More efficient implementation using a min-heap to track driver availability.
    """
    if not deliveries:
        return {}
    
    # Sort deliveries by start time
    sorted_deliveries = sorted(deliveries, key=lambda x: x['start'])
    
    # Min-heap of (driver_available_time, driver_id)
    driver_heap = []
    assignments = {}  # driver_id -> list of deliveries
    next_driver_id = 0
    
    for delivery in sorted_deliveries:
        start = delivery['start']
        end = delivery['end']
        
        # Check if any driver is available
        if driver_heap and driver_heap[0][0] <= start:
            # Reuse an existing driver
            available_time, driver_id = heapq.heappop(driver_heap)
            heapq.heappush(driver_heap, (end, driver_id))
            assignments[driver_id].append(delivery)
        else:
            # Need a new driver
            driver_id = next_driver_id
            next_driver_id += 1
            heapq.heappush(driver_heap, (end, driver_id))
            assignments[driver_id] = [delivery]
    
    return assignments


def test_package_prioritization():
    """Test the maximize_deliveries function."""
    print("Testing Package Prioritization...")
    
    test_cases = [
        # Basic test
        {
            'windows': [
                {'start': 9.0, 'end': 10.0},
                {'start': 10.0, 'end': 11.0},
                {'start': 9.5, 'end': 10.5}
            ],
            'expected_count': 2
        },
        # Empty list
        {
            'windows': [],
            'expected_count': 0
        },
        # Overlapping deliveries
        {
            'windows': [
                {'start': 9.0, 'end': 11.0},
                {'start': 10.0, 'end': 12.0},
                {'start': 11.0, 'end': 13.0},
                {'start': 12.0, 'end': 14.0}
            ],
            'expected_count': 2
        },
        # Non-overlapping deliveries
        {
            'windows': [
                {'start': 9.0, 'end': 10.0},
                {'start': 10.0, 'end': 11.0},
                {'start': 11.0, 'end': 12.0}
            ],
            'expected_count': 3
        }
    ]
    
    for i, test in enumerate(test_cases):
        result = maximize_deliveries(test['windows'])
        print(f"  Test {i+1}: Expected {test['expected_count']}, Got {len(result)}")
        if len(result) != test['expected_count']:
            print(f"    WARNING: Test {i+1} failed!")
    
    print("Package Prioritization tests complete.\n")

def test_truck_loading():
    """Test the optimize_truck_load function."""
    print("Testing Truck Loading...")
    
    test_cases = [
        # Basic test
        {
            'packages': [
                {'weight': 10, 'value': 60},
                {'weight': 20, 'value': 100},
                {'weight': 30, 'value': 120}
            ],
            'weight_limit': 50,
            'expected_value': 240  # Take all: 60+100+120=280? Wait, 50 limit means not all
            # Actually: 60+100=160 + 2/3 of 120=80, total 240
        },
        # Empty list
        {
            'packages': [],
            'weight_limit': 50,
            'expected_value': 0
        },
        # Zero weight limit
        {
            'packages': [
                {'weight': 10, 'value': 60}
            ],
            'weight_limit': 0,
            'expected_value': 0
        }
    ]
    
    for i, test in enumerate(test_cases):
        result = optimize_truck_load(test['packages'], test['weight_limit'])
        print(f"  Test {i+1}: Expected value {test['expected_value']}, Got {result['total_value']:.1f}")
        if abs(result['total_value'] - test['expected_value']) > 0.01:
            print(f"    WARNING: Test {i+1} failed!")
    
    print("Truck Loading tests complete.\n")

def test_driver_assignment():
    """Test the minimize_drivers function."""
    print("Testing Driver Assignment...")
    
    test_cases = [
        # Basic test - should need 2 drivers
        {
            'deliveries': [
                {'start': 9.0, 'end': 11.0},
                {'start': 10.0, 'end': 12.0},
                {'start': 11.0, 'end': 13.0}
            ],
            'expected_drivers': 2
        },
        # All non-overlapping - should need 1 driver
        {
            'deliveries': [
                {'start': 9.0, 'end': 10.0},
                {'start': 10.0, 'end': 11.0},
                {'start': 11.0, 'end': 12.0}
            ],
            'expected_drivers': 1
        },
        # All overlapping - should need 3 drivers
        {
            'deliveries': [
                {'start': 9.0, 'end': 11.0},
                {'start': 9.0, 'end': 11.0},
                {'start': 9.0, 'end': 11.0}
            ],
            'expected_drivers': 3
        }
    ]
    
    for i, test in enumerate(test_cases):
        result = minimize_drivers(test['deliveries'])
        print(f"  Test {i+1}: Expected {test['expected_drivers']} drivers, Got {len(result)}")
        if len(result) != test['expected_drivers']:
            print(f"    WARNING: Test {i+1} failed!")
    
    print("Driver Assignment tests complete.\n")


def benchmark_scenarios():
    """Run all implementations on the generated scenarios and time them."""
    print("=" * 50)
    print("PERFORMANCE BENCHMARKING")
    print("=" * 50)
    
    scenarios = ['small_scenario.json', 'medium_scenario.json', 'large_scenario.json']
    
    for scenario_file in scenarios:
        print(f"\nBenchmarking {scenario_file}...")
        data = load_scenario(scenario_file)
        
        # Part 1: Package Prioritization
        start = time.time()
        result1 = maximize_deliveries(data['time_windows'])
        time1 = time.time() - start
        print(f"  Package Prioritization: {len(result1)} deliveries scheduled in {time1:.4f}s")
        
        # Part 2: Truck Loading
        start = time.time()
        result2 = optimize_truck_load(data['packages'], data['weight_limit'])
        time2 = time.time() - start
        print(f"  Truck Loading: {result2['total_value']:.1f} value achieved in {time2:.4f}s")
        
        # Part 3: Driver Assignment
        start = time.time()
        result3 = minimize_drivers(data['deliveries'])
        time3 = time.time() - start
        print(f"  Driver Assignment: {len(result3)} drivers needed in {time3:.4f}s")


if __name__ == "__main__":
    print("GREEDY OPTIMIZER FOR DELIVERY ROUTING")
    print("=" * 50)
    
    # Run tests
    test_package_prioritization()
    test_truck_loading()
    test_driver_assignment()
    
    # Uncomment to run benchmarks
    # benchmark_scenarios()
    
    print("\nAll tests complete! Uncomment benchmark_scenarios() to run performance tests.")