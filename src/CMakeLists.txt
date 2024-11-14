add_executable(main main.cpp)

target_link_libraries(main PUBLIC benchmark::benchmark mdspan)

add_sycl_to_target(TARGET main)
