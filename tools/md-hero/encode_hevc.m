// Encode frame_####.png (RGBA) into a QuickTime movie with HEVC + alpha, the only
// transparent video Safari and iOS play. Unlike avconvert's presets, the bitrate is
// set explicitly, which keeps a 1080 px loop inside the site's ~3 MB budget.
//   clang -fobjc-arc -framework Foundation -framework AVFoundation \
//       -framework CoreMedia -framework CoreVideo -framework VideoToolbox \
//       -framework ImageIO -framework CoreGraphics -framework Accelerate \
//       tools/md-hero/encode_hevc.m -o /tmp/encode_hevc
//   /tmp/encode_hevc FRAMES_DIR OUT.mov [FPS=30] [KBPS=1600] [ALPHA_QUALITY=0.75]
// KBPS sets the color layer; the alpha layer comes on top, sized by ALPHA_QUALITY.
#import <Accelerate/Accelerate.h>
#import <AVFoundation/AVFoundation.h>
#import <ImageIO/ImageIO.h>
#import <VideoToolbox/VideoToolbox.h>

// Draw a PNG into a BGRA pixel buffer as straight (unpremultiplied) alpha, which is
// what Apple's own HEVC-with-alpha presets write.
static BOOL fill(CVPixelBufferRef pb, NSURL *url) {
    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
    CGImageRef img = src ? CGImageSourceCreateImageAtIndex(src, 0, NULL) : NULL;
    if (src) CFRelease(src);
    if (!img) return NO;
    CVPixelBufferLockBaseAddress(pb, 0);
    size_t w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb);
    size_t stride = CVPixelBufferGetBytesPerRow(pb);
    void *base = CVPixelBufferGetBaseAddress(pb);
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(base, w, h, 8, stride, cs,
        kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little);
    CGContextClearRect(ctx, CGRectMake(0, 0, w, h));
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
    vImage_Buffer buf = { base, h, w, stride };
    vImageUnpremultiplyData_BGRA8888(&buf, &buf, kvImageNoFlags);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CGImageRelease(img);
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: encode_hevc FRAMES_DIR OUT.mov [FPS=30] [KBPS=1600]"
                            " [ALPHA_QUALITY=0.75]\n");
            return 2;
        }
        NSString *dir = @(argv[1]);
        NSURL *out = [NSURL fileURLWithPath:@(argv[2])];
        int fps = argc > 3 ? atoi(argv[3]) : 30;
        int kbps = argc > 4 ? atoi(argv[4]) : 1600;
        double alphaQuality = argc > 5 ? atof(argv[5]) : 0.75;
        NSPredicate *isFrame = [NSPredicate predicateWithFormat:
            @"SELF BEGINSWITH 'frame_' AND SELF ENDSWITH '.png'"];
        NSArray<NSString *> *names = [[[NSFileManager.defaultManager
            contentsOfDirectoryAtPath:dir error:nil] filteredArrayUsingPredicate:isFrame]
            sortedArrayUsingSelector:@selector(compare:)];
        if (names.count == 0) { fprintf(stderr, "no frame_*.png in %s\n", argv[1]); return 1; }

        NSURL *first = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:names[0]]];
        CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)first, NULL);
        NSDictionary *props = CFBridgingRelease(CGImageSourceCopyPropertiesAtIndex(src, 0, NULL));
        CFRelease(src);
        int w = [props[(id)kCGImagePropertyPixelWidth] intValue];
        int h = [props[(id)kCGImagePropertyPixelHeight] intValue];

        [NSFileManager.defaultManager removeItemAtURL:out error:nil];
        NSError *err = nil;
        AVAssetWriter *writer = [AVAssetWriter assetWriterWithURL:out
                                                         fileType:AVFileTypeQuickTimeMovie
                                                            error:&err];
        if (!writer) { fprintf(stderr, "%s\n", err.description.UTF8String); return 1; }
        AVAssetWriterInput *input = [AVAssetWriterInput
            assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:@{
                AVVideoCodecKey: AVVideoCodecTypeHEVCWithAlpha,
                AVVideoWidthKey: @(w),
                AVVideoHeightKey: @(h),
                AVVideoCompressionPropertiesKey: @{
                    AVVideoAverageBitRateKey: @(kbps * 1000),
                    AVVideoExpectedSourceFrameRateKey: @(fps),
                    AVVideoMaxKeyFrameIntervalKey: @(fps * 2),
                    (__bridge NSString *)kVTCompressionPropertyKey_TargetQualityForAlpha:
                        @(alphaQuality),
                },
            }];
        input.expectsMediaDataInRealTime = NO;
        AVAssetWriterInputPixelBufferAdaptor *adaptor = [AVAssetWriterInputPixelBufferAdaptor
            assetWriterInputPixelBufferAdaptorWithAssetWriterInput:input
            sourcePixelBufferAttributes:@{
                (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (id)kCVPixelBufferWidthKey: @(w),
                (id)kCVPixelBufferHeightKey: @(h),
            }];
        [writer addInput:input];
        if (![writer startWriting]) {
            fprintf(stderr, "%s\n", writer.error.description.UTF8String);
            return 1;
        }
        [writer startSessionAtSourceTime:kCMTimeZero];

        for (NSUInteger i = 0; i < names.count; i++) {
            @autoreleasepool {
                while (!input.readyForMoreMediaData) [NSThread sleepForTimeInterval:0.005];
                CVPixelBufferRef pb = NULL;
                CVPixelBufferPoolCreatePixelBuffer(NULL, adaptor.pixelBufferPool, &pb);
                NSURL *url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:names[i]]];
                if (!pb || !fill(pb, url)) {
                    fprintf(stderr, "cannot read %s\n", names[i].UTF8String);
                    return 1;
                }
                CVBufferSetAttachment(pb, kCVImageBufferAlphaChannelModeKey,
                                      kCVImageBufferAlphaChannelMode_StraightAlpha,
                                      kCVAttachmentMode_ShouldPropagate);
                BOOL ok = [adaptor appendPixelBuffer:pb
                                withPresentationTime:CMTimeMake((int64_t)i, fps)];
                CVPixelBufferRelease(pb);
                if (!ok) { fprintf(stderr, "%s\n", writer.error.description.UTF8String); return 1; }
            }
        }
        [input markAsFinished];
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [writer finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(done); }];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (writer.status != AVAssetWriterStatusCompleted) {
            fprintf(stderr, "%s\n", writer.error.description.UTF8String);
            return 1;
        }
        printf("wrote %s: %lu frames, %dx%d, %d fps, %d kbps target\n",
               argv[2], (unsigned long)names.count, w, h, fps, kbps);
    }
    return 0;
}
