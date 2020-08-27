pragma Ada_2012;
pragma Style_Checks (Off);
pragma Warnings ("U");

with Interfaces.C; use Interfaces.C;
with Interfaces.C.Extensions;

package uvector_types_h is

   type char1 is record
      x : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:100
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:98

   type uchar1 is record
      x : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:105
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:103

   type char2 is record
      x : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:111
      y : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:111
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:109

   type uchar2 is record
      x : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:116
      y : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:116
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:114

   type char3 is record
      x : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:121
      y : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:121
      z : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:121
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:119

   type uchar3 is record
      x : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:126
      y : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:126
      z : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:126
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:124

   type char4 is record
      x : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:131
      y : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:131
      z : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:131
      w : aliased signed_char;  -- /usr/local/cuda/include//vector_types.h:131
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:129

   type uchar4 is record
      x : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:136
      y : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:136
      z : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:136
      w : aliased unsigned_char;  -- /usr/local/cuda/include//vector_types.h:136
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:134

   type short1 is record
      x : aliased short;  -- /usr/local/cuda/include//vector_types.h:141
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:139

   type ushort1 is record
      x : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:146
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:144

   type short2 is record
      x : aliased short;  -- /usr/local/cuda/include//vector_types.h:151
      y : aliased short;  -- /usr/local/cuda/include//vector_types.h:151
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:149

   type ushort2 is record
      x : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:156
      y : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:156
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:154

   type short3 is record
      x : aliased short;  -- /usr/local/cuda/include//vector_types.h:161
      y : aliased short;  -- /usr/local/cuda/include//vector_types.h:161
      z : aliased short;  -- /usr/local/cuda/include//vector_types.h:161
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:159

   type ushort3 is record
      x : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:166
      y : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:166
      z : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:166
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:164

   type short4 is record
      x : aliased short;  -- /usr/local/cuda/include//vector_types.h:169
      y : aliased short;  -- /usr/local/cuda/include//vector_types.h:169
      z : aliased short;  -- /usr/local/cuda/include//vector_types.h:169
      w : aliased short;  -- /usr/local/cuda/include//vector_types.h:169
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:169

   type ushort4 is record
      x : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:170
      y : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:170
      z : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:170
      w : aliased unsigned_short;  -- /usr/local/cuda/include//vector_types.h:170
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:170

   type int1 is record
      x : aliased int;  -- /usr/local/cuda/include//vector_types.h:174
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:172

   type uint1 is record
      x : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:179
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:177

   type int2 is record
      x : aliased int;  -- /usr/local/cuda/include//vector_types.h:182
      y : aliased int;  -- /usr/local/cuda/include//vector_types.h:182
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:182

   type uint2 is record
      x : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:183
      y : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:183
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:183

   type int3 is record
      x : aliased int;  -- /usr/local/cuda/include//vector_types.h:187
      y : aliased int;  -- /usr/local/cuda/include//vector_types.h:187
      z : aliased int;  -- /usr/local/cuda/include//vector_types.h:187
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:185

   type uint3 is record
      x : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:192
      y : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:192
      z : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:192
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:190

   type int4 is record
      x : aliased int;  -- /usr/local/cuda/include//vector_types.h:197
      y : aliased int;  -- /usr/local/cuda/include//vector_types.h:197
      z : aliased int;  -- /usr/local/cuda/include//vector_types.h:197
      w : aliased int;  -- /usr/local/cuda/include//vector_types.h:197
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:195

   type uint4 is record
      x : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:202
      y : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:202
      z : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:202
      w : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:202
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:200

   type long1 is record
      x : aliased long;  -- /usr/local/cuda/include//vector_types.h:207
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:205

   type ulong1 is record
      x : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:212
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:210

   type long2 is record
      x : aliased long;  -- /usr/local/cuda/include//vector_types.h:222
      y : aliased long;  -- /usr/local/cuda/include//vector_types.h:222
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:220

   type ulong2 is record
      x : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:227
      y : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:227
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:225

   type long3 is record
      x : aliased long;  -- /usr/local/cuda/include//vector_types.h:234
      y : aliased long;  -- /usr/local/cuda/include//vector_types.h:234
      z : aliased long;  -- /usr/local/cuda/include//vector_types.h:234
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:232

   type ulong3 is record
      x : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:239
      y : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:239
      z : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:239
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:237

   type long4 is record
      x : aliased long;  -- /usr/local/cuda/include//vector_types.h:244
      y : aliased long;  -- /usr/local/cuda/include//vector_types.h:244
      z : aliased long;  -- /usr/local/cuda/include//vector_types.h:244
      w : aliased long;  -- /usr/local/cuda/include//vector_types.h:244
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:242

   type ulong4 is record
      x : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:249
      y : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:249
      z : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:249
      w : aliased unsigned_long;  -- /usr/local/cuda/include//vector_types.h:249
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:247

   type float1 is record
      x : aliased float;  -- /usr/local/cuda/include//vector_types.h:254
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:252

   type float2 is record
      x : aliased float;  -- /usr/local/cuda/include//vector_types.h:274
      y : aliased float;  -- /usr/local/cuda/include//vector_types.h:274
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:274

   type float3 is record
      x : aliased float;  -- /usr/local/cuda/include//vector_types.h:281
      y : aliased float;  -- /usr/local/cuda/include//vector_types.h:281
      z : aliased float;  -- /usr/local/cuda/include//vector_types.h:281
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:279

   type float4 is record
      x : aliased float;  -- /usr/local/cuda/include//vector_types.h:286
      y : aliased float;  -- /usr/local/cuda/include//vector_types.h:286
      z : aliased float;  -- /usr/local/cuda/include//vector_types.h:286
      w : aliased float;  -- /usr/local/cuda/include//vector_types.h:286
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:284

   type longlong1 is record
      x : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:291
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:289

   type ulonglong1 is record
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:296
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:294

   type longlong2 is record
      x : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:301
      y : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:301
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:299

   type ulonglong2 is record
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:306
      y : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:306
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:304

   type longlong3 is record
      x : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:311
      y : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:311
      z : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:311
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:309

   type ulonglong3 is record
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:316
      y : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:316
      z : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:316
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:314

   type longlong4 is record
      x : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:321
      y : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:321
      z : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:321
      w : aliased Long_Long_Integer;  -- /usr/local/cuda/include//vector_types.h:321
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:319

   type ulonglong4 is record
      x : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:326
      y : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:326
      z : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:326
      w : aliased Extensions.unsigned_long_long;  -- /usr/local/cuda/include//vector_types.h:326
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:324

   type double1 is record
      x : aliased double;  -- /usr/local/cuda/include//vector_types.h:331
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:329

   type double2 is record
      x : aliased double;  -- /usr/local/cuda/include//vector_types.h:336
      y : aliased double;  -- /usr/local/cuda/include//vector_types.h:336
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:334

   type double3 is record
      x : aliased double;  -- /usr/local/cuda/include//vector_types.h:341
      y : aliased double;  -- /usr/local/cuda/include//vector_types.h:341
      z : aliased double;  -- /usr/local/cuda/include//vector_types.h:341
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:339

   type double4 is record
      x : aliased double;  -- /usr/local/cuda/include//vector_types.h:346
      y : aliased double;  -- /usr/local/cuda/include//vector_types.h:346
      z : aliased double;  -- /usr/local/cuda/include//vector_types.h:346
      w : aliased double;  -- /usr/local/cuda/include//vector_types.h:346
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:344

   type dim3 is record
      x : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:418
      y : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:418
      z : aliased unsigned;  -- /usr/local/cuda/include//vector_types.h:418
   end record
   with Convention => C_Pass_By_Copy;  -- /usr/local/cuda/include//vector_types.h:416

end uvector_types_h;
