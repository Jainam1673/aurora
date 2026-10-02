#include "aurora/aurora.hpp"

#include <gtest/gtest.h>

TEST(AuroraSmokeTest, VersionIntegrity) {
    EXPECT_EQ(aurora::kVersion, "0.1.0");
    EXPECT_EQ(aurora::kVersionMajor, 0);
    EXPECT_EQ(aurora::kVersionMinor, 1);
    EXPECT_EQ(aurora::kVersionPatch, 0);
}

TEST(AuroraSmokeTest, DescriptionNonEmpty) {
    auto desc = aurora::description();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("AURORA"), std::string_view::npos);
}

TEST(AuroraSmokeTest, SystemInfoValid) {
    auto info = aurora::get_system_info();
    EXPECT_EQ(info.version, "0.1.0");
    EXPECT_FALSE(info.cpp_standard.empty());
    EXPECT_FALSE(info.compiler.empty());

    // Verify C++23 standard (__cplusplus >= 202302L)
    long cpp_val = std::stol(info.cpp_standard);
    EXPECT_GE(cpp_val, 202302L);
}
