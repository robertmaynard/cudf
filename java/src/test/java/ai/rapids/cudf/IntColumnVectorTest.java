/*
 *  Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.Builder;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class IntColumnVectorTest extends CudfTestBase {

  @Test
  public void testCreateColumnVectorBuilder() {
    try (ColumnVector intColumnVector = ColumnVector.build(DType.INT32, 3, (b) -> b.append(1))) {
      assertFalse(intColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    try (HostColumnVector intColumnVector = HostColumnVector.fromInts(2, 3, 5)) {
      assertFalse(intColumnVector.hasNulls());
      assertEquals(intColumnVector.getInt(0), 2);
      assertEquals(intColumnVector.getInt(1), 3);
      assertEquals(intColumnVector.getInt(2), 5);
    }
  }

  @Test
  public void testUnsignedArrayAllocation() {
    try (HostColumnVector v = HostColumnVector.fromUnsignedInts(0xfedcba98, 0x80000000, 5)) {
      assertFalse(v.hasNulls());
      assertEquals(0xfedcba98L, Integer.toUnsignedLong(v.getInt(0)));
      assertEquals(0x80000000L, Integer.toUnsignedLong(v.getInt(1)));
      assertEquals(5, Integer.toUnsignedLong(v.getInt(2)));
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    try (HostColumnVector intColumnVector = HostColumnVector.fromInts(2, 3, 5)) {
      assertThrows(AssertionError.class, () -> intColumnVector.getInt(3));
      assertFalse(intColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    try (HostColumnVector intColumnVector = HostColumnVector.fromInts(2, 3, 5)) {
      assertFalse(intColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> intColumnVector.getInt(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    try (HostColumnVector cv = HostColumnVector.fromBoxedInts(2, 3, 4, 5, 6, 7, null, null)) {
      assertTrue(cv.hasNulls());
      assertEquals(2, cv.getNullCount());
      for (int i = 0; i < 6; i++) {
        assertFalse(cv.isNull(i));
      }
      assertTrue(cv.isNull(6));
      assertTrue(cv.isNull(7));
    }
  }

  @Test
  public void testAddingUnsignedNullValues() {
    try (HostColumnVector cv = HostColumnVector.fromBoxedUnsignedInts(
            2, 3, 4, 5, 0xfedbca98, 0x80000000, null, null)) {
      assertTrue(cv.hasNulls());
      assertEquals(2, cv.getNullCount());
      for (int i = 0; i < 6; i++) {
        assertFalse(cv.isNull(i));
      }
      assertEquals(0xfedbca98L, Integer.toUnsignedLong(cv.getInt(4)));
      assertEquals(0x80000000L, Integer.toUnsignedLong(cv.getInt(5)));
      assertTrue(cv.isNull(6));
      assertTrue(cv.isNull(7));
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (Builder builder = HostColumnVector.builder(DType.INT32, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(2).appendNull().appendArray(new int[]{5, 4}).build());
    }
  }

  @Test
  public void testCastToInt() {
    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(new double[]{4.3, 3.8, 8});
         ColumnVector shortColumnVector = ColumnVector.fromShorts(new short[]{100});
         ColumnVector intColumnVector1 = doubleColumnVector.asInts();
         ColumnVector expected1 = ColumnVector.fromInts(4, 3, 8);
         ColumnVector intColumnVector2 = shortColumnVector.asInts();
         ColumnVector expected2 = ColumnVector.fromInts(100)) {
      AssertUtils.assertColumnsAreEqual(expected1, intColumnVector1);
      AssertUtils.assertColumnsAreEqual(expected2, intColumnVector2);
    }
  }

  @Test
  void testAppendVector() {
    Random random = new Random(192312989128L);
    for (int dstSize = 1; dstSize <= 100; dstSize++) {
      for (int dstPrefilledSize = 0; dstPrefilledSize < dstSize; dstPrefilledSize++) {
        final int srcSize = dstSize - dstPrefilledSize;
        for (int sizeOfDataNotToAdd = 0; sizeOfDataNotToAdd <= dstPrefilledSize; sizeOfDataNotToAdd++) {
          try (Builder dst = HostColumnVector.builder(DType.INT32, dstSize);
               HostColumnVector src = HostColumnVector.build(DType.INT32, srcSize, (b) -> {
                 for (int i = 0; i < srcSize; i++) {
                   if (random.nextBoolean()) {
                     b.appendNull();
                   } else {
                     b.append(random.nextInt());
                   }
                 }
               });
               Builder gtBuilder = HostColumnVector.builder(DType.INT32,
                   dstPrefilledSize)) {
            assertEquals(dstSize, srcSize + dstPrefilledSize);
            //add the first half of the prefilled list
            for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
              if (random.nextBoolean()) {
                dst.appendNull();
                gtBuilder.appendNull();
              } else {
                int a = random.nextInt();
                dst.append(a);
                gtBuilder.append(a);
              }
            }
            // append the src vector
            dst.append(src);
            try (HostColumnVector dstVector = dst.build();
                 HostColumnVector gt = gtBuilder.build()) {
              for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
                assertEquals(gt.isNull(i), dstVector.isNull(i));
                if (!gt.isNull(i)) {
                  assertEquals(gt.getInt(i), dstVector.getInt(i));
                }
              }
              for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                assertEquals(src.isNull(j), dstVector.isNull(i));
                if (!src.isNull(j)) {
                  assertEquals(src.getInt(j), dstVector.getInt(i));
                }
              }
              if (dstVector.hasValidityVector()) {
                long maxIndex =
                    BitVectorHelper.getValidityAllocationSizeInBytes(dstVector.getRowCount()) * 8;
                for (long i = dstSize - sizeOfDataNotToAdd; i < maxIndex; i++) {
                  assertFalse(dstVector.isNullExtendedRange(i));
                }
              }
            }
          }
        }
      }
    }
  }
}
