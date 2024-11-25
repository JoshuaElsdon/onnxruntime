﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Based on https://github.com/mattleibow/DeviceRunners/tree/main/sample/SampleMauiApp

using DeviceRunners.VisualRunners;
using DeviceRunners.VisualRunners.Xunit;
using Microsoft.Extensions.Logging;

#if MODE_XHARNESS
using DeviceRunners.XHarness;
#endif

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;

/// <summary>
/// ITestRunner that starts/stops Appium server before/after running tests.
/// </summary>
public class XunitTestRunnerWithAppium : ITestRunner
{
    private readonly XunitTestRunner runner;

    public XunitTestRunnerWithAppium(IVisualTestRunnerConfiguration options, 
                                     IResultChannelManager? resultChannelManager = null, 
                                     IDiagnosticsManager? diagnosticsManager = null)
    {
        runner = new XunitTestRunner(options, resultChannelManager, diagnosticsManager);
    }

    public Task RunTestsAsync(IEnumerable<ITestAssemblyInfo> testAssemblies, 
                              CancellationToken cancellationToken = default(CancellationToken))
    {
        return RunTests(runner.RunTestsAsync(testAssemblies, cancellationToken));
    }

    public Task RunTestsAsync(IEnumerable<ITestCaseInfo> testCases, 
                              CancellationToken cancellationToken = default(CancellationToken))
    {
        return RunTests(runner.RunTestsAsync(testCases, cancellationToken));
    }

    private static async Task StartAppiumServer()
    {
        await Task.Run(() => System.Console.WriteLine("Starting Appium Server"));
    }

    private static async Task StopAppiumServer()
    {
        await Task.Run(() => System.Console.WriteLine("Stopping Appium Server"));
    }

    private static async Task RunTests(Task testTask)
    {
        await StartAppiumServer();
        await testTask;
        await StopAppiumServer();
    }
}

/// <summary>
/// Class that can capture test results.
/// </summary>
public class InternalCaptureResultChannel : IResultChannel
{
    static readonly char[] NewLineChars = ['\r', '\n'];
    readonly object locker = new(); // not sure if this is needed. copied from DeviceRunners.VisualRunners usage

    int failed = 0;
    int passed = 0;
    int skipped = 0;
    SortedDictionary<string, ITestResultInfo> failedTests = new();
    bool running = false;

    public bool IsOpen => running;

    public Task<bool> OpenChannel(string? message = null)
    {
        lock (locker)
        {
            // we could hook in Appium start here but it feels a little nicer to do it at the ITestRunner level

            // reset counters and any failed test info
            failed = passed = skipped = 0;
            failedTests.Clear();

            running = true;
        }

        return Task.FromResult(true);
    }

    public Task CloseChannel()
    {
        lock (locker)
        {
            running = false;
        }

        return Task.CompletedTask;
    }

    public void RecordResult(ITestResultInfo result)
    {
        var name = result.TestCase.DisplayName;

        switch (result.Status)
        {
            case TestResultStatus.Passed:
                passed++;
                return;
            case TestResultStatus.Skipped:
                skipped++;
                return;
            case TestResultStatus.Failed:
                failed++;
                failedTests.Add(name, result);
                break;
        }

        // error message and stack trace can be found/processed with the below
        var message = result.ErrorMessage;
        if (!string.IsNullOrEmpty(message))
        {
        }

        var stacktrace = result.ErrorStackTrace;
        if (!string.IsNullOrEmpty(stacktrace))
        {
            var lines = stacktrace.Split(NewLineChars, StringSplitOptions.RemoveEmptyEntries);
            foreach (var line in lines)
            {
                // do something
            }
        }
    }
}

public static class VisualTestRunnerConfigurationBuilderExtensions2
{
    public static TBuilder AddXunitWithAppium<TBuilder>(this TBuilder builder) 
        where TBuilder : IVisualTestRunnerConfigurationBuilder
    {
        builder.AddTestPlatform<XunitTestDiscoverer, XunitTestRunnerWithAppium>();
        return builder;
    }

    public static TBuilder AddInternalCaptureResultChannel<TBuilder>(this TBuilder builder)
        where TBuilder : IVisualTestRunnerConfigurationBuilder
    {
        builder.AddResultChannel(_ => new InternalCaptureResultChannel());
        return builder;
    }
}

public static class MauiProgram
{
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder
            // .ConfigureUITesting()
#if MODE_XHARNESS
            .UseXHarnessTestRunner(conf => conf
                .AddTestAssembly(typeof(MauiProgram).Assembly)
                .AddXunit())
#endif
            .UseVisualTestRunner(conf => conf
//#if MODE_NON_INTERACTIVE_VISUAL
//                .EnableAutoStart(true)
//                .AddTcpResultChannel(new TcpResultChannelOptions
//                {
//                    HostNames = ["localhost", "10.0.2.2"],
//                    Port = 16384,
//                    Formatter = new TextResultChannelFormatter(),
//                    Required = false
//                }) 
//#endif
                .AddConsoleResultChannel()
                .AddInternalCaptureResultChannel()
                .AddTestAssembly(typeof(MauiProgram).Assembly)
                .AddXunitWithAppium()
                .EnableAutoStart()
                );

#if DEBUG
        builder.Logging.AddDebug();
#endif

        return builder.Build();
    }
}
