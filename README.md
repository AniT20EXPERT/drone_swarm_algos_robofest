# Drone_swarm_algos_robofest


THE FOLLOWING DIRECTORIES IN THIS REPO ARE AS FOLLOWS:

PIPELINE_A: MAIN DRONE STITCHING + DETECTION + PATH PLANNING 
1) image capture + stitching
2) running yolo + sahi on stitched image for detection
3) get path planning [as a list of coords in image pixels] [so some mapping between drone x,y,z and the actual coords in form of pixels in stitched image]

PIPELINE_B: ADVANCED SPEECH TO TEXT WITH WAKE WORD + USER COMMAND IDENTIFICATION + CONFIRMATION + EXECUTION SERVICE/PARAM CALLS + TEXT TO SPEECH 
1) porcupine wake word detection
2) silero-vad Voice activity detection
3) vosk speech to text
4) intent analysis
5) text to speech for declaration of user command, pending confirmation
6) silero-vad + vosk: to get user response[yes/no]
7) user response is passed to intent analysis to finally get confirmation/rejection of command
8) command execution + command message via text to speech

[NOTE: THE FOLLOWING YOLO_PREP CAN BE SKIPPED ITS FOR INFO ONLY, I HAVE ALREADY PROVIDED A TRAINED CUSTOM YOLO MODEL IN REPO, SO USE IT DIRECTLY IN THE ABOVE PIPELINE_A] :-

YOLO_PREP: THE CUSTOM SYNTHETIC IMAGE DATASET GENERATION SCRIPT AND THE YOLO TRAINING SCRIPT 
1) synthetic image generation script
2) training yolo script
3) testing and running inference script




