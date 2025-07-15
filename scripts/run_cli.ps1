# scripts\run_cli.ps1  ─── run four trackers on one clip
$VID  = "..\2min.mp4"
$YOLO = "..\weights\yolov8x_person_face.pt"
$REID = "..\weights\reid\osnet_x1_0_msmt17.pt"   # omit for ByteTrack

foreach ($TRK in "bytetrack","botsort","deepocsort","boosttrack") {
    python -m boxmot.track `
        --source          $VID `
        --yolo-model      $YOLO `
        --tracking-method $TRK `
        --reid-model      $REID `
        --project         "..\results" `
        --name            $TRK `
        --save-vid --save-txt --verbose
}
