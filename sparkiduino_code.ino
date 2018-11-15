#include <sparki.h>
#define TRUE 1
#define FALSE 0

// States
#define STATE_GET_COMMAND 1 // recieve and process bluetooth command of which line to travel to
#define STATE_START 2 // path planning using dijsktra
#define STATE_HAS_PATH 3 // figure out where we currently are and where we want to travel
#define STATE_SEEKING_POSE 4 // moving to the next pose
#define STATE_IN_POSITION 5 // reached the destination pose (line to shoot from) and make final angle adjustments (from bluetooth)
#define STATE_SHOOT 6 // trigger motor/catapult assembly
#define STATE_IDLE 7

// Robot constants
#define DISTANCE_MARGIN 0.02 // 2cm of tolerance
#define HEADING_MARGIN 0.061 // ~3.5 degrees of tolerance
#define ROBOT_SPEED 0.0278
#define CYCLE_TIME .100
#define AXLE_DIAMETER 0.0865
#define M_PI 3.14159
#define WHEEL_RADIUS 0.03
#define FWD 1
#define NONE 0
#define BCK -1

#define INITIAL_GOAL_STATE_I -1
#define INITIAL_GOAL_STATE_J -1

// Number of vertices to discretize the map
#define NUM_X_CELLS 6
#define NUM_Y_CELLS 6

// Map is ~42cm x 42cm
#define MAP_SIZE_X 0.42
#define MAP_SIZE_Y 0.42

#define BIG_NUMBER 255

int current_state = STATE_GET_COMMAND; // Change this variable to determine which controller to run

short prev_holder[NUM_X_CELLS*NUM_Y_CELLS];

// IK/Odometry Variables
float pose_x = 0., pose_y = 0., pose_theta = 0.;
float dest_pose_x = 0., dest_pose_y = 0., dest_pose_theta = 0.;
float d_err = 0., b_err = 0., h_err = 0., phi_l = 0., phi_r = 0.;

// Wheel rotation vars
float left_speed_pct = 0.;
float right_speed_pct = 0.;
int left_dir = DIR_CCW;
int right_dir = DIR_CW;
int left_wheel_rotating = NONE;
int right_wheel_rotating = NONE;

//float dX  = 0., dTheta = 0.;

// Temporary goal variables
float next_x = -1;
float next_y = -1;
float next_theta = 0;

bool world_map[NUM_Y_CELLS][NUM_X_CELLS];

// Example source/dest pair: Move from (0,0) to (2,1)
bool goal_changed = TRUE; // Track if the goal coordinates have been changed since we last path planned.
int goal_i = INITIAL_GOAL_STATE_I;
int goal_j = INITIAL_GOAL_STATE_J;
int source_i = 0;
int source_j = 0;
short *prev = NULL;
short *path = NULL;

long program_start_time = 0; // Track time since controller began

void setup() {

  delay(5000); //Wait 5 seconds
  
  // IK Setup
  Serial.begin(9600);
  pose_x = 0.;
  pose_y = 0.;
  pose_theta = 0.;
  left_wheel_rotating = NONE;
  right_wheel_rotating = NONE;

  set_pose_destination(0, 0, 0);

  // Dijkstra Setup
  for (int j = 0; j < NUM_Y_CELLS; ++j) {
    for (int i = 0; i < NUM_X_CELLS; ++i) {
      world_map[j][i] = 1;
    }
  }
  
  
  //SETTING THE OBSTACLES (INCLUDE OBSTACLE CELLS FROM OUR COMPUTER VISION MANUALLY)
  // world_map[i][j] = 0;
  world_map[1][4] = 0;
  world_map[3][2] = 0;
    
  program_start_time = millis();
}

/*****************************
 * IK Helper Functions       *
 *****************************/
void set_pose_destination(float x, float y, float t) {
  dest_pose_x = x;
  dest_pose_y = y;
  dest_pose_theta = t;
  if (dest_pose_theta > M_PI) dest_pose_theta -= 2*M_PI;
  if (dest_pose_theta < -M_PI) dest_pose_theta += 2*M_PI;
}

float to_radians(double deg) {
  return  deg * 3.1415/180.;
}

float to_degrees(double rad) {
  return  rad * 180 / 3.1415;
}

void updateOdometry() {
  pose_x += cos(pose_theta) * CYCLE_TIME * ROBOT_SPEED 
         * (float(left_wheel_rotating)*left_speed_pct 
            + float(right_wheel_rotating)*right_speed_pct)/2.;
  pose_y += sin(pose_theta) * CYCLE_TIME * ROBOT_SPEED 
         * (float(left_wheel_rotating)*left_speed_pct 
            + float(right_wheel_rotating)*right_speed_pct)/2.;
  pose_theta += (float(right_wheel_rotating)*right_speed_pct 
                 - float(left_wheel_rotating)*left_speed_pct)  
              * CYCLE_TIME * ROBOT_SPEED / AXLE_DIAMETER;
  if (pose_theta > M_PI) pose_theta -= 2.*M_PI;
  if (pose_theta <= -M_PI) pose_theta += 2.*M_PI;
}

/*****************************
 * Core IK Functions         *
 *****************************/

 
void moveStop() {
  left_wheel_rotating = NONE;
  right_wheel_rotating = NONE;
  left_speed_pct = 0.0;
  right_speed_pct = 0.0;
  sparki.moveStop();
}
void compute_IK_errors() {
  // Distance, Bearing, and Heading error
  d_err = sqrt( (dest_pose_x-pose_x)*(dest_pose_x-pose_x) + (dest_pose_y-pose_y)*(dest_pose_y-pose_y) );
  b_err = atan2( (dest_pose_y - pose_y), (dest_pose_x - pose_x) ) - pose_theta;
  h_err = dest_pose_theta - pose_theta;
  
  if (b_err <= -M_PI) b_err += 2.*M_PI;
  else if (b_err > M_PI) b_err -= 2.*M_PI;
  
  if (h_err <= -M_PI) h_err += 2.*M_PI;
  else if (h_err > M_PI) h_err -= 2.*M_PI;  
}

void compute_IK_wheel_rotations() {
  float dTheta, dX;
  if (d_err > DISTANCE_MARGIN) { // Get reasonably close before considering heading error
    dTheta = b_err;
  } else {
    dTheta = h_err;
    dX = 0.; // Force 0-update for dX since we're supposedly at the goal position anyway
  }  

  dX  = 0.1 * min(M_PI, d_err); // De-prioritize distance error to help avoid paths through unintended grid cells
  
  phi_l = (dX  - (dTheta*AXLE_DIAMETER/2.)) / WHEEL_RADIUS;
  phi_r = (dX  + (dTheta*AXLE_DIAMETER/2.)) / WHEEL_RADIUS;
}

void set_IK_motor_rotations() {
  float wheel_rotation_normalizer = max(abs(phi_l), abs(phi_r));
  if (wheel_rotation_normalizer == 0) { moveStop(); return; }
  left_speed_pct = abs(phi_l) / wheel_rotation_normalizer;
  right_speed_pct = abs(phi_r) / wheel_rotation_normalizer;

  // Figure out which direction the wheels need to rotate
  if (phi_l <= -to_radians(1)) {
    left_dir = DIR_CW;
    left_wheel_rotating = BCK;
  } else if (phi_l >= to_radians(1)) {
    left_dir = DIR_CCW;
    left_wheel_rotating = FWD;
  } else {
    left_speed_pct = 0.;
    left_wheel_rotating = 0;
  }

  if (phi_r <= -to_radians(1)) {
    right_dir = DIR_CCW;
    right_wheel_rotating = BCK;
  } else if (phi_r >= to_radians(1)) {
    right_dir = DIR_CW;
    right_wheel_rotating = FWD;
  } else {
    right_speed_pct = 0.;
    right_wheel_rotating = 0;
  }

  sparki.motorRotate(MOTOR_LEFT, left_dir, int(left_speed_pct*100));
  sparki.motorRotate(MOTOR_RIGHT, right_dir, int(right_speed_pct*100));
}

bool is_robot_at_IK_destination_pose() {
  return (d_err <= DISTANCE_MARGIN && abs(h_err) < HEADING_MARGIN);
}



/*****************************
 * Dijkstra Helper Functions *
 *****************************/

// Return 1 if there are entries in range [0,inf) in arr
// otherwise return 0, signifying empty queue
bool is_not_empty(short *arr, int len) {
  for (int i=0; i < len; ++i) {
    if (arr[i] >= 0) {
      return TRUE;
    }
  }
  return FALSE;
}

int get_min_index(short *arr, int len) {
  int min_idx=0;
  for (int i=0;i < len; ++i) {
    if (arr[min_idx] < 0 || (arr[i] < arr[min_idx] && arr[i] >= 0)) {
      min_idx = i;
    }
  }
  if (arr[min_idx] == -1) return -1; 
  return min_idx;
}


/**********************************
 * Coordinate Transform Functions *
 **********************************/

// Returns 0 if something went wrong -- assume invalid i and j values being set
bool vertex_index_to_ij_coordinates(int v_idx, int *i, int *j) {
  *i = v_idx % NUM_X_CELLS;
  *j = v_idx / NUM_X_CELLS;
  
  if (*i < 0 || *j < 0 || *i >= NUM_X_CELLS || *j >= NUM_Y_CELLS) return FALSE;
  return TRUE;
}

int ij_coordinates_to_vertex_index(int i, int j) {
  return j*NUM_X_CELLS + i;  
}

// Returns 0 if something went wrong -- assume invalid x and y values are being set
// Returns 1 otherwise. x and y values are the middle of cell (i,j)
bool ij_coordinates_to_xy_coordinates(int i, int j, float *x, float *y) {
  if (i < 0 || j < 0 || i >= NUM_X_CELLS || j >= NUM_Y_CELLS) return FALSE;
  
  *x = (i+0.5)*(MAP_SIZE_X/NUM_X_CELLS);
  *y = (j+0.5)*(MAP_SIZE_Y/NUM_Y_CELLS);
  return TRUE;  
}

// Returns 0 if something went wrong -- assume invalid x and y values are being set
// Returns 1 otherwise. x and y values are the middle of cell (i,j)
bool xy_coordinates_to_ij_coordinates(float x, float y, int *i, int *j) {
  if (x < 0 || y < 0 || x >= MAP_SIZE_X || y >= MAP_SIZE_Y) return FALSE;
  
  *i = int((x/MAP_SIZE_X) * NUM_X_CELLS);
  *j = int((y/MAP_SIZE_Y) * NUM_Y_CELLS);

  return TRUE;  
}

/**********************************
 *      Core Dijkstra Functions   *
 **********************************/


// Returns the cost of moving from vertex_source to vertex_dest
byte get_travel_cost(int vertex_source, int vertex_dest) {
  bool are_neighboring = 1;

  int s_i, s_j; // Source i,j
  int d_i, d_j; // Dest i,j

  if (!vertex_index_to_ij_coordinates(vertex_source, &s_i, &s_j)) return BIG_NUMBER;
  if (!vertex_index_to_ij_coordinates(vertex_dest, &d_i, &d_j)) return BIG_NUMBER;

  are_neighboring = (abs(s_i - d_i) + abs(s_j - d_j) <= 1); // 4-Connected world

  // TODO: Add your code to incorporate world_map here --> DONE
  if (world_map[s_i][s_j] == 0){
    return BIG_NUMBER;
  }
  if (world_map[d_i][d_j] == 0){
    return BIG_NUMBER;
  }

  if (are_neighboring)
    return 1;
  else
    return BIG_NUMBER;
}

short *run_dijkstra(bool world_map[NUM_Y_CELLS][NUM_X_CELLS], int source_vertex) {
  int cur_vertex = -1;
  int cur_cost = -1;
  int min_index = -1;
  int travel_cost = -1;
  byte dist[NUM_X_CELLS*NUM_Y_CELLS];
  short Q_cost[NUM_Y_CELLS*NUM_X_CELLS];
  short *prev = prev_holder;

  // Initialize our variables
  for (int i = 0; i < NUM_X_CELLS * NUM_Y_CELLS; ++i) {
    Q_cost[i] = -1;
    dist[i] = BIG_NUMBER;
    prev[i] = -1;
  }

  dist[source_vertex] = 0;
  Q_cost[source_vertex] = 0;

  while (is_not_empty(Q_cost,NUM_X_CELLS*NUM_Y_CELLS)) {
    min_index = get_min_index(Q_cost, NUM_X_CELLS*NUM_Y_CELLS);
    if (min_index < 0) {
      break; // Queue is empty somehow!
    }
    cur_vertex = min_index; // Vertex ID is same as array indices
    cur_cost = Q_cost[cur_vertex]; // Current best cost for reaching vertex
    Q_cost[cur_vertex] = -1; // Remove cur_vertex from the queue.

    // Iterate through all of node's neighbors and see if cur_vertex provides a shorter
    // path to the neighbor than whatever route may exist currently.
    for (byte neighbor_idx = 0; neighbor_idx < NUM_X_CELLS*NUM_Y_CELLS; ++neighbor_idx) {     
      short alt = -1;
      travel_cost = get_travel_cost(cur_vertex, neighbor_idx);
      if (travel_cost == BIG_NUMBER) {
        continue; // Nodes are not neighbors/cannot traverse... skip!
      }
      alt = dist[cur_vertex] + travel_cost;
      if (alt < dist[neighbor_idx]) {
        // New shortest path to node at neighbor_idx found!
        dist[neighbor_idx] = alt;
        Q_cost[neighbor_idx] = alt;
        prev[neighbor_idx] = cur_vertex;
      }
    } 
  }
  return prev;
}

short *reconstruct_path(short *prev, int source_vertex, int dest_vertex) {
  short path[NUM_X_CELLS * NUM_Y_CELLS]; // Max-length that path could be
  short *final_path = NULL;
  int last_idx = 0;

  path[last_idx++] = dest_vertex; // Start at goal, work backwards
  short last_vertex = prev[dest_vertex];
  while (last_vertex != -1) {
    path[last_idx++] = last_vertex;
    last_vertex = prev[last_vertex];
  }

  final_path = new short[last_idx+1];
  for (int i = 0; i < last_idx; ++i) {
    final_path[i] = path[last_idx-1-i]; // Reverse path so it goes source->dest
  }
  final_path[last_idx] = -1; // End-of-array marker
  return final_path;
}





void displayOdometry() {
  sparki.clearLCD();
  sparki.print("X: ");
  sparki.print(pose_x);
  sparki.print(" Xg: ");
  sparki.println(dest_pose_x);
  sparki.print("Y: ");
  sparki.print(pose_y);
  sparki.print(" Yg: ");
  sparki.println(dest_pose_y); 
  sparki.print("T: ");
  sparki.print(pose_theta*180./M_PI);
  sparki.print(" Tg: ");
  sparki.println(dest_pose_theta*180./M_PI);

//  sparki.print("dX : ");
//  sparki.print(dX );
//  sparki.print("   dT: ");
//  sparki.println(dTheta);
  sparki.print("phl: "); sparki.print(phi_l); sparki.print(" phr: "); sparki.println(phi_r);
  sparki.print("p: "); sparki.print(d_err); sparki.print(" a: "); sparki.println(to_degrees(b_err));
  sparki.print("h: "); sparki.println(to_degrees(h_err));  
  sparki.updateLCD();
}

void loop () {
  unsigned long begin_time = millis();
  unsigned long end_time = 0;

  updateOdometry();
  displayOdometry();
  // Your code should work with this test uncommented!
  /*
  if (millis() - program_start_time > 15000 && goal_i != 0 && goal_j != 0) {
    // After 15 seconds of operation, set the goal vertex to 0,0!
    goal_i = 0; goal_j = 0;    
    goal_changed = TRUE;
  }
  */

  /****************************************/
  // Implement your state machine here    //
  // in place of the example code         //
  /****************************************/

  if (goal_changed){ //if we need to recalculate a path to new goal
    current_state = STATE_START;    
  }
  
  Serial.println(current_state);
  switch(current_state){

     
      // state 1
      case STATE_GET_COMMAND:{           
          // Possible goals (i,j)
          // Line 3: (3,0) or (3,6)
          // Line 2: (4,1) or (4,5)
          // Line 1: (5,2) or (5,4)

          Serial.println("test");
          
          int dest_line = 3; // eventually get this from bluetooth
          if (dest_line == 3){
            goal_i = 3; 
          }
          if (dest_line == 2){
            goal_i = 4; 
          }
          if (dest_line == 1){
            goal_i = 5; 
          }
          int choose_corner = random(0,2);
          if (choose_corner == 0){
            if (dest_line == 3){
              goal_j = 0; 
            }
            if (dest_line == 2){
              goal_j = 1; 
            }
            if (dest_line == 1){
              goal_j = 2; 
            }
          }            
          if (choose_corner == 1){
            if (dest_line == 3){
              goal_j = 6; 
            }
            if (dest_line == 2){
              goal_j = 5; 
            }
            if (dest_line == 1){
              goal_j = 4; 
            }
          }
      }
      break;
      // state 2
      case STATE_START: {
          //get current robot i and j
          xy_coordinates_to_ij_coordinates(pose_x, pose_y, &source_i, &source_j);
          delete path; path=NULL; 
          // Example code to retrieve a path from Dijkstra //
            prev = run_dijkstra(world_map, ij_coordinates_to_vertex_index(source_i, source_j)); 
            path = reconstruct_path(prev, ij_coordinates_to_vertex_index(source_i, source_j), ij_coordinates_to_vertex_index(goal_i, goal_j));
            goal_changed = FALSE;
            
            if (path[0] == ij_coordinates_to_vertex_index(source_i, source_j) ){
              //we succesfully found a path
              current_state = STATE_HAS_PATH;
              
              int idx = 0;
              Serial.println("path");
              while (path[idx] != -1){
                
                
                Serial.println(path[idx]);
                idx++;
              }
            }
            
      }
      

      // state 3
      case STATE_HAS_PATH:{
               int idx = 0;
              Serial.println("path");
              while (path[idx] != -1){
                
                
                Serial.println(path[idx]);
                idx++;
              }
          //get current robot i and j
          xy_coordinates_to_ij_coordinates(pose_x, pose_y, &source_i, &source_j);
          int curr_index = ij_coordinates_to_vertex_index(source_i, source_j);
          Serial.println("curr");
          Serial.println(curr_index);
          int next_vertex = -2;
          int next_next_vertex = -3;
          
          //find where we are in the path
          idx = 0;
          
          while (path[idx] != -1){
              if (curr_index == path[idx] ){
                  //this is where we currently are in path
                  next_vertex = path[idx+1]; //this is where we want to go next move
                  Serial.println("next");
                  Serial.println(next_vertex);
                  next_next_vertex = path[idx+2]; //this is where we want to go in two moves                       
                  
              }
              idx++;
          }
          if (next_vertex == -1){ //if we are already at the goal vertex...
                      moveStop();
                      current_state = STATE_IDLE;
                      break;
           }   
          int next_i = -1;
          int next_j = -1;

          int next_next_i = -1;
          int next_next_j = -1;
          float next_next_x = -1;
          float next_next_y = -1;
          
          // Finding next vertex
          vertex_index_to_ij_coordinates(next_vertex, &next_i, &next_j);
          ij_coordinates_to_xy_coordinates(next_i, next_j, &next_x, &next_y); //now we have the goal x-y coordinates
          
          
          if (next_next_vertex == -1){ //if it doesn't matter where we face at the end...
              next_theta = 0;
          }          
          else{
              // Finding coordinates of next next vertex
              vertex_index_to_ij_coordinates(next_next_vertex, &next_next_i, &next_next_j);
              ij_coordinates_to_xy_coordinates(next_next_i, next_next_j, &next_next_x, &next_next_y); //now we have the goal x-y coordinates
              next_theta = atan2(next_next_y - next_y, next_next_x - next_x);
          }
          
          //we have the x,y,theta to move to, so now go to state 3
          set_pose_destination(next_x,next_y,next_theta);
          current_state = STATE_SEEKING_POSE;
      }
          break;

      //state 4
      case STATE_SEEKING_POSE:{
      
      
            // Example code to use IK //
            compute_IK_errors();
            compute_IK_wheel_rotations();
            set_IK_motor_rotations();
            
            if (is_robot_at_IK_destination_pose()) {
              moveStop();
              current_state=STATE_HAS_PATH;
              break;
            }
      }
      break;
      // state 5
      case STATE_IN_POSITION:{
        
      }
      break;
      // state 6
      case STATE_SHOOT:{
        
      }
      break;
      //idle state 7
      case STATE_IDLE:{
        
      }
      break;

 }


  ///////////////////////////

  // TODO: Do something with path here instead of just displaying it!
  
  sparki.clearLCD();
  sparki.print("Source: "); sparki.print(source_i); sparki.print(", "); sparki.println(source_j);
  sparki.print("Goal: "); sparki.print(goal_i); sparki.print(", "); sparki.println(goal_j);
  for (int i=0; path[i] != -1; ++i) {
    sparki.print(path[i]);
    sparki.print(" -> "); 
  }
  sparki.println(" DONE! "); 
  sparki.updateLCD();
  
  //delete path; path=NULL; // Important! Delete the arrays returned from reconstruct_path when you're done with them!
 
  ///////////////////////////////////////////////////  
 
  end_time = millis();
  if (end_time - begin_time < 1000*CYCLE_TIME)
    delay(1000*CYCLE_TIME - (end_time - begin_time)); // each loop takes CYCLE_TIME ms
  else
    delay(10); // Accept some error
}

