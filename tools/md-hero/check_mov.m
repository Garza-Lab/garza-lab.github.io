// Print codec, alpha, size and duration of a movie's first video track.
//   clang -fobjc-arc -framework Foundation -framework AVFoundation \
//       -framework CoreMedia tools/md-hero/check_mov.m -o /tmp/check_mov
//   /tmp/check_mov uploads/hero/e6-hero.mov
#import <AVFoundation/AVFoundation.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: check_mov FILE\n"); return 2; }
        NSURL *url = [NSURL fileURLWithPath:@(argv[1])];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
        AVAssetTrack *track = [[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
        if (!track || track.formatDescriptions.count == 0) { printf("no video track\n"); return 1; }
        CMFormatDescriptionRef desc = (__bridge CMFormatDescriptionRef)track.formatDescriptions[0];
        FourCharCode sub = CMFormatDescriptionGetMediaSubType(desc);
        CFTypeRef alpha = CMFormatDescriptionGetExtension(
            desc, kCMFormatDescriptionExtension_ContainsAlphaChannel);
        CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(desc);
        printf("codec=%c%c%c%c alpha=%s size=%dx%d seconds=%.2f\n",
               (char)(sub >> 24), (char)(sub >> 16), (char)(sub >> 8), (char)sub,
               alpha == kCFBooleanTrue ? "true" : "false",
               dims.width, dims.height, CMTimeGetSeconds(asset.duration));
    }
    return 0;
}
