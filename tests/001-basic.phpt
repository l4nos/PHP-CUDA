--TEST--
Basic CUDA functionality test
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
?>
--FILE--
<?php
// Test device count
$count = cuda_device_count();
var_dump($count > 0);

// Test device properties
if ($count > 0) {
    $props = cuda_device_properties(0);
    var_dump(isset($props['name']));
    var_dump(isset($props['totalGlobalMem']));
    var_dump(isset($props['maxThreadsPerBlock']));
}

// Test matrix multiplication
$matrix_a = [
    [1.0, 2.0],
    [3.0, 4.0]
];

$matrix_b = [
    [5.0, 6.0],
    [7.0, 8.0]
];

$result = [];
$success = cuda_matrix_multiply($matrix_a, $matrix_b, $result);
var_dump($success);

// Expected result:
// [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
// [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
var_dump($result);

// Test error handling
$last_error = cuda_get_last_error();
var_dump($last_error === 0); // 0 means cudaSuccess

// Test invalid matrix multiplication (should fail gracefully)
$invalid_matrix = [
    [1.0, 2.0],
    [3.0] // Inconsistent row length
];
$result = [];
$success = cuda_matrix_multiply($invalid_matrix, $matrix_b, $result);
var_dump($success === false);

?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(19)
    [1]=>
    float(22)
  }
  [1]=>
  array(2) {
    [0]=>
    float(43)
    [1]=>
    float(50)
  }
}
bool(true)
bool(true)
